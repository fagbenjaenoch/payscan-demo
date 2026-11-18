package com.example.payscandemo

import android.Manifest
import android.app.Activity
import android.content.ClipData
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
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
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.core.ZoomState
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.Observer
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.time.Duration.Companion.milliseconds

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Minimum gap between OCR runs while actively searching for an account number. */
private const val ACTIVE_ANALYSIS_INTERVAL_MS = 400L

/** Slower polling interval once a HIGH-confidence NUBAN is locked to reduce thermal/battery load. */
private const val LOCKED_ANALYSIS_INTERVAL_MS = 900L

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

// ---------------------------------------------------------------------------
// Permission handling
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// OCR analyzer (delegates NUBAN filtering to NubanFilter on workerExecutor)
// ---------------------------------------------------------------------------

/** One recognized text block. [box] is in preview-view pixels, or null if it couldn't be mapped. */
data class OcrBlock(
    val text: String,
    val box: Rect?,
    val matchesNuban: Boolean = false
)

/** Combined frame output containing raw OCR blocks and any confirmed NUBAN candidate. */
data class OcrScanResult(
    val blocks: List<OcrBlock>,
    val nuban: NubanCandidate?
)

class OcrAnalyzer(
    private val recognizer: TextRecognizer,
    private val nubanFilter: NubanFilter,
    private val workerExecutor: Executor,
    private val isPaused: () -> Boolean,
    private val isLocked: () -> Boolean,
    private val viewSize: () -> IntSize,
    private val onResult: (OcrScanResult) -> Unit
) : ImageAnalysis.Analyzer {

    private val released = AtomicBoolean(false)
    private var lastRunAt = 0L

    /** Call before closing the recognizer so late frames/results are ignored safely. */
    fun release() = released.set(true)

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        val mediaImage = imageProxy.image
        val minInterval = if (isLocked()) LOCKED_ANALYSIS_INTERVAL_MS else ACTIVE_ANALYSIS_INTERVAL_MS

        // Throttle, pause and shutdown checks. Always close the proxy or the stream freezes.
        if (released.get() || isPaused() || mediaImage == null ||
            now - lastRunAt < minInterval
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
                .addOnSuccessListener(workerExecutor) { visionText ->
                    if (released.get()) return@addOnSuccessListener
                    val size = viewSize()
                    val confirmedNuban = nubanFilter.filter(visionText.text)

                    val textBlocks = visionText.textBlocks
                    val blocks = ArrayList<OcrBlock>(textBlocks.size)
                    for (i in 0 until textBlocks.size) {
                        val block = textBlocks[i]
                        val text = block.text
                        if (text.isNotBlank()) {
                            val trimmed = text.trim()
                            val box = block.boundingBox?.let { mapToView(it, imageW, imageH, size) }
                            val matchesNuban = confirmedNuban != null &&
                                    nubanFilter.matchesBlock(trimmed, confirmedNuban)
                            blocks.add(OcrBlock(trimmed, box, matchesNuban))
                        }
                    }

                    if (blocks.isNotEmpty() || confirmedNuban != null) {
                        onResult(OcrScanResult(blocks, confirmedNuban))
                    }
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

private val aspectStrategy = AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY

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
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

// ---------------------------------------------------------------------------
// Main screen
// ---------------------------------------------------------------------------

@Composable
fun CameraWithTorchAndOcrScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val clipboard = LocalClipboard.current
    val scope = rememberCoroutineScope()

    val nubanFilter = remember { NubanFilter() }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var isTorchOn by rememberSaveable { mutableStateOf(false) }
    var zoomRatio by rememberSaveable { mutableFloatStateOf(1f) }
    var minZoomRatio by remember { mutableFloatStateOf(1f) }
    var maxZoomRatio by remember { mutableFloatStateOf(1f) }
    var blocks by remember { mutableStateOf<List<OcrBlock>>(emptyList()) }
    var detectedNuban by remember { mutableStateOf<NubanCandidate?>(null) }
    var lastTextAt by remember { mutableLongStateOf(0L) }
    var isFrozen by remember { mutableStateOf(false) }
    var frozenFrame by remember { mutableStateOf<ImageBitmap?>(null) }
    var selectedIndex by remember { mutableStateOf<Int?>(null) }
    var previewSize by remember { mutableStateOf(IntSize.Zero) }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    DisposableEffect(lifecycleOwner) {
        val executor = Executors.newSingleThreadExecutor()
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        val analyzer = OcrAnalyzer(
            recognizer = recognizer,
            nubanFilter = nubanFilter,
            workerExecutor = executor,
            isPaused = { isFrozen },
            isLocked = { detectedNuban?.confidence == Confidence.HIGH },
            viewSize = { previewSize },
            onResult = { result ->
                if (!isFrozen) {
                    if (result.blocks.isNotEmpty()) {
                        blocks = result.blocks
                        lastTextAt = SystemClock.elapsedRealtime()
                    }
                    if (result.nuban != null) {
                        detectedNuban = result.nuban
                        lastTextAt = SystemClock.elapsedRealtime()
                    }
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

    // Torch: re-applied whenever the camera rebinds or the saved state changes.
    LaunchedEffect(camera, isTorchOn) {
        camera?.takeIf { it.cameraInfo.hasFlashUnit() }
            ?.cameraControl?.enableTorch(isTorchOn)
    }

    DisposableEffect(camera, lifecycleOwner) {
        val cam = camera
        val observer = Observer<ZoomState> { state ->
            minZoomRatio = state.minZoomRatio
            maxZoomRatio = state.maxZoomRatio
            zoomRatio = zoomRatio.coerceIn(minZoomRatio, maxZoomRatio)
        }
        cam?.cameraInfo?.zoomState?.observe(lifecycleOwner, observer)

        onDispose {
            cam?.cameraInfo?.zoomState?.removeObserver(observer)
        }
    }

    LaunchedEffect(camera, zoomRatio, minZoomRatio, maxZoomRatio) {
        camera?.cameraControl?.setZoomRatio(zoomRatio.coerceIn(minZoomRatio, maxZoomRatio))
    }

    // Clear stale text and NUBAN votes after the camera points away.
    LaunchedEffect(Unit) {
        while (true) {
            delay(500.milliseconds)
            if (!isFrozen && (blocks.isNotEmpty() || detectedNuban != null) &&
                SystemClock.elapsedRealtime() - lastTextAt > STALE_TEXT_MS
            ) {
                blocks = emptyList()
                detectedNuban = null
                nubanFilter.reset()
            }
        }
    }

    fun toggleFreeze() {
        if (isFrozen) {
            isFrozen = false
            frozenFrame = null
            selectedIndex = null
            blocks = emptyList()
            detectedNuban = null
            nubanFilter.reset()
        } else if (blocks.isNotEmpty() || detectedNuban != null) {
            frozenFrame = previewView.bitmap?.asImageBitmap()
            isFrozen = true
        }
    }

    fun copyText(label: String, text: String) {
        scope.launch {
            clipboard.setClipEntry(ClipEntry(ClipData.newPlainText(label, text)))
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
                Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val selectedBlock = selectedIndex?.let { blocks.getOrNull(it) }
    val displayText = remember(selectedBlock, blocks) {
        selectedBlock?.text ?: blocks.joinToString("\n") { it.text }
    }
    val hasFlash = camera?.cameraInfo?.hasFlashUnit() == true

    Box(
        modifier = Modifier
            .fillMaxSize()
            .then(
                if (!isFrozen) {
                    Modifier.pointerInput(camera, minZoomRatio, maxZoomRatio) {
                        detectTransformGestures { _, _, zoomChange, _ ->
                            if (!isFrozen) {
                                zoomRatio = (zoomRatio * zoomChange).coerceIn(minZoomRatio, maxZoomRatio)
                            }
                        }
                    }
                } else {
                    Modifier
                }
            )
    ) {
        AndroidView(
            modifier = Modifier
                .fillMaxSize()
                .onSizeChanged { previewSize = it },
            factory = { previewView }
        )

        // Frozen snapshot covers the live preview so boxes stay aligned with the image.
        frozenFrame?.let { frame ->
            Image(
                bitmap = frame,
                contentDescription = null,
                contentScale = ContentScale.FillBounds,
                modifier = Modifier.fillMaxSize()
            )
        }

        // Single-pass Canvas for bounding boxes; NUBAN-matching blocks highlight in green, tapped in yellow.
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .then(
                    if (isFrozen) {
                        Modifier.pointerInput(blocks) {
                            detectTapGestures { tapOffset ->
                                val hit = blocks.indexOfLast { it.box?.contains(tapOffset) == true }
                                selectedIndex = when {
                                    hit == -1 || hit == selectedIndex -> null
                                    else -> hit
                                }
                            }
                        }
                    } else {
                        Modifier
                    }
                )
        ) {
            val selectedStroke = 3.dp.toPx()
            val defaultStroke = 1.5.dp.toPx()
            val defaultColor = Color.White.copy(alpha = 0.8f)
            val nubanColor = Color(0xFF00E676)

            blocks.forEachIndexed { index, block ->
                val r = block.box ?: return@forEachIndexed
                val selected = index == selectedIndex
                val color = when {
                    selected -> Color.Yellow
                    block.matchesNuban -> nubanColor
                    else -> defaultColor
                }
                drawRect(
                    color = color,
                    topLeft = r.topLeft,
                    size = r.size,
                    style = Stroke(
                        width = if (selected || block.matchesNuban) selectedStroke else defaultStroke
                    )
                )
            }
        }

        // Bottom overlay: detected NUBAN card + raw OCR text + actions
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            detectedNuban?.let { candidate ->
                NubanResultCard(
                    candidate = candidate,
                    onCopyAccount = { copyText("NUBAN account number", it) }
                )
            }

            Surface(
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                shape = MaterialTheme.shapes.medium,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = if (detectedNuban != null) 110.dp else 150.dp)
            ) {
                Text(
                    text = when {
                        displayText.isNotEmpty() -> displayText
                        else -> "Point the camera at a bank account or text"
                    },
                    modifier = Modifier
                        .padding(12.dp)
                        .verticalScroll(rememberScrollState()),
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            if (isFrozen) {
                Text(
                    text = "Tap a box to copy just that block",
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.White
                )
            }

            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (maxZoomRatio > 1f) {
                    OutlinedButton(
                        enabled = !isFrozen,
                        onClick = {
                            zoomRatio = if (zoomRatio < 1.8f) {
                                2f.coerceAtMost(maxZoomRatio)
                            } else {
                                1f
                            }
                        }
                    ) {
                        Text(String.format(Locale.US, "%.1fx", zoomRatio))
                    }
                }

                if (hasFlash) {
                    Button(onClick = { isTorchOn = !isTorchOn }) {
                        Text(if (isTorchOn) "Flash Off" else "Flash On")
                    }
                }

                Button(
                    enabled = isFrozen || blocks.isNotEmpty() || detectedNuban != null,
                    onClick = { toggleFreeze() }
                ) {
                    Text(if (isFrozen) "Resume" else "Freeze")
                }

                Button(
                    enabled = displayText.isNotEmpty(),
                    onClick = { copyText("Recognized text", displayText) }
                ) {
                    Text(if (selectedBlock != null) "Copy Block" else "Copy All")
                }
            }
        }
    }
}

@Composable
private fun NubanResultCard(
    candidate: NubanCandidate,
    onCopyAccount: (String) -> Unit
) {
    val (badgeBg, badgeFg) = when (candidate.confidence) {
        Confidence.HIGH -> Color(0xFF1B5E20) to Color(0xFFA5D6A7)
        Confidence.MEDIUM -> Color(0xFFE65100) to Color(0xFFFFCC80)
        Confidence.LOW -> MaterialTheme.colorScheme.errorContainer to MaterialTheme.colorScheme.onErrorContainer
    }

    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.95f),
        shape = MaterialTheme.shapes.medium,
        tonalElevation = 4.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = candidate.bank?.let { "${it.name} (${it.code})" } ?: "Bank not detected",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = "${candidate.confidence.name} CONFIDENCE",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = badgeFg,
                    modifier = Modifier
                        .background(badgeBg, RoundedCornerShape(4.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = candidate.number,
                    style = MaterialTheme.typography.headlineSmall.copy(
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    ),
                    color = MaterialTheme.colorScheme.onSecondaryContainer
                )
                Button(onClick = { onCopyAccount(candidate.number) }) {
                    Text("Copy NUBAN")
                }
            }

            candidate.suggestedFix?.let { fix ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Possible OCR typo → $fix",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedButton(onClick = { onCopyAccount(fix) }) {
                        Text("Copy Fix")
                    }
                }
            }
        }
    }
}