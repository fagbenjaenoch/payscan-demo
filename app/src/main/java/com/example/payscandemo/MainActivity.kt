package com.example.payscandemo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Minimum gap between OCR runs. Saves battery and keeps the phone cool. */
private const val ANALYSIS_INTERVAL_MS = 400L

/** Clear live results if no text has been seen for this long. */
private const val STALE_TEXT_MS = 1_500L

/** Target analysis resolution. Higher than CameraX's 640x480 default so small print is readable. */
private val ANALYSIS_SIZE = Size(1280, 720)
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                CameraFlashlightApp()
            }
        }
    }
}

private fun Context.findActivity(): Activity? {
    var ctx = this
    while (ctx is ContextWrapper) {
        if (ctx is Activity) return ctx
        ctx = ctx.baseContext
    }
    return null
}

@Composable
fun CameraFlashlightApp() {
    val context = LocalContext.current
    val activity = remember(context) { context.findActivity() }

    fun isGranted() = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    var hasCameraPermission by remember { mutableStateOf(isGranted()) }
    var permanentlyDenied by remember { mutableStateOf(false) }
    var askedOnce by rememberSaveable { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        permanentlyDenied = !granted && activity != null &&
                !ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)
    }

    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        hasCameraPermission = isGranted()
        if (hasCameraPermission) permanentlyDenied = false
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission && !askedOnce) {
            askedOnce = true
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (hasCameraPermission) {
        CameraWithTorchAndOcrScreen()
    } else {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = if (permanentlyDenied) {
                    "Camera access is turned off. Enable it in Settings to scan text."
                } else {
                    "Camera access is needed to scan text."
                },
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyLarge
            )
            Spacer(Modifier.height(16.dp))
            if (permanentlyDenied) {
                Button(onClick = {
                    context.startActivity(
                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                            data = Uri.fromParts("package", context.packageName, null)
                        }
                    )
                }) { Text("Open Settings") }
            } else {
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Grant Camera Permission")
                }
            }
        }
    }
}

private val aspectStrategy = AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY

// ---------------------------------------------------------------------------
// OCR analyzer (kept out of the composable so it stays small and testable)
// ---------------------------------------------------------------------------

/** One recognized text block. [box] is in preview-view pixels, or null if it couldn't be mapped. */
data class OcrBlock(val text: String, val box: Rect?)

class OcrAnalyzer(
    private val recognizer: TextRecognizer,
    private val isPaused: () -> Boolean,
    private val viewSize: () -> IntSize,
    /** Always invoked on the main thread. */
    private val onResult: (List<OcrBlock>) -> Unit
) : ImageAnalysis.Analyzer {

    private val released = AtomicBoolean(false)
    private var lastRunAt = 0L

    /** Call before closing the recognizer so late frames/results are ignored safely. */
    fun release() = released.set(true)

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        val mediaImage = imageProxy.image

        // Throttle, pause and shutdown checks. Always close the proxy or the stream freezes.
        if (released.get() || isPaused() || mediaImage == null ||
            now - lastRunAt < ANALYSIS_INTERVAL_MS
        ) {
            imageProxy.close()
            return
        }
        lastRunAt = now

        val rotation = imageProxy.imageInfo.rotationDegrees
        val upright = rotation % 180 == 0
        val imageW = if (upright) imageProxy.width else imageProxy.height
        val imageH = if (upright) imageProxy.height else imageProxy.width
        val image = InputImage.fromMediaImage(mediaImage, rotation)

        try {
            recognizer.process(image)
                .addOnSuccessListener { visionText ->
                    if (released.get()) return@addOnSuccessListener
                    val size = viewSize()
                    val blocks = visionText.textBlocks
                        .filter { it.text.isNotBlank() }
                        .map { block ->
                            OcrBlock(
                                text = block.text.trim(),
                                box = block.boundingBox?.let { mapToView(it, imageW, imageH, size) }
                            )
                        }
                    onResult(blocks)
                }
                .addOnCompleteListener { imageProxy.close() }
        } catch (_: Exception) {
            // Recognizer already closed or similar: never leak the frame.
            imageProxy.close()
        }
    }

    /** Maps image coordinates to PreviewView coordinates (FILL_CENTER, same aspect ratio on both use cases). */
    private fun mapToView(box: android.graphics.Rect, imageW: Int, imageH: Int, view: IntSize): Rect? {
        if (view.width == 0 || view.height == 0 || imageW == 0 || imageH == 0) return null
        val scale = max(view.width / imageW.toFloat(), view.height / imageH.toFloat())
        val dx = (view.width - imageW * scale) / 2f
        val dy = (view.height - imageH * scale) / 2f
        return Rect(
            left = box.left * scale + dx,
            top = box.top * scale + dy,
            right = box.right * scale + dx,
            bottom = box.bottom * scale + dy
        )
    }
}

// ---------------------------------------------------------------------------
// Camera use case builders
// ---------------------------------------------------------------------------

private fun buildPreview(previewView: PreviewView): Preview =
    Preview.Builder()
        .setResolutionSelector(
            ResolutionSelector.Builder().setAspectRatioStrategy(aspectStrategy).build()
        )
        .build()
        .also { it.surfaceProvider = previewView.surfaceProvider }

private fun buildImageAnalysis(): ImageAnalysis =
    ImageAnalysis.Builder()
        .setResolutionSelector(
            ResolutionSelector.Builder()
                .setAspectRatioStrategy(aspectStrategy)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        ANALYSIS_SIZE,
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()
        )
        // Drop frames while ML Kit is busy so the preview never lags
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

@Composable
fun CameraWithTorchAndOcrScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val clipboardManager = LocalClipboardManager.current

    var camera by remember { mutableStateOf<Camera?>(null) }
    var isTorchOn by remember { mutableStateOf(false) }
    var detectedText by remember { mutableStateOf("") }
    var previewSize by remember { mutableStateOf(IntSize.Zero) }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    DisposableEffect(lifecycleOwner) {
        val executor = Executors.newSingleThreadExecutor()
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        val analyzer = OcrAnalyzer(
            recognizer = recognizer,
            isPaused = { false },
            viewSize = { previewSize },
            onResult = { result ->
                if (result.isNotEmpty()) {
                    detectedText = result.joinToString("\n") { it.text }
                }
            }
        )
        val imageAnalysis = buildImageAnalysis().also { it.setAnalyzer(executor, analyzer) }

        val providerFuture = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        var disposed = false

        providerFuture.addListener({
            if (disposed) return@addListener
            val p = providerFuture.get()
            provider = p
            p.unbindAll()
            camera = p.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                buildPreview(previewView),
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            disposed = true
            analyzer.release()
            imageAnalysis.clearAnalyzer()
            provider?.unbindAll()
            executor.shutdown()
            recognizer.close()
            camera = null
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier
                .fillMaxSize()
                .onSizeChanged { previewSize = it },
            factory = { previewView }
        )

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (detectedText.isNotEmpty()) {
                Surface(
                    color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                    shape = MaterialTheme.shapes.medium,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 150.dp)
                ) {
                    Text(
                        text = detectedText,
                        modifier = Modifier
                            .padding(12.dp)
                            .verticalScroll(rememberScrollState()),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = {
                        val cam = camera ?: return@Button
                        if (cam.cameraInfo.hasFlashUnit()) {
                            isTorchOn = !isTorchOn
                            cam.cameraControl.enableTorch(isTorchOn)
                        }
                    }
                ) {
                    Text(if (isTorchOn) "Flash Off" else "Flash On")
                }

                Button(
                    enabled = detectedText.isNotEmpty(),
                    onClick = {
                        clipboardManager.setText(AnnotatedString(detectedText))
                        Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                    }
                ) {
                    Text("Copy Text")
                }
            }
        }
    }
}