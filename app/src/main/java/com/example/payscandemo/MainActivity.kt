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
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountBalance
import androidx.compose.material.icons.filled.AcUnit
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
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

    // Torch: re-applied whenever the camera rebinds or the saved state changes.
    LaunchedEffect(camera, isTorchOn) {
        camera?.takeIf { it.cameraInfo.hasFlashUnit() }
            ?.cameraControl?.enableTorch(isTorchOn)
    }

    // Re-apply zoom ratio on camera rebind or zoomRatio change
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
            .background(Color.Black)
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
        // Centered 4:3 Camera Feed Container (Restricted to middle portion of screen)
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .fillMaxWidth(0.92f)
                .aspectRatio(3f / 4f)
                .clip(RoundedCornerShape(16.dp))
                .background(Color.DarkGray)
                .onSizeChanged { previewSize = it }
        ) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { previewView }
            )

            // Frozen snapshot covers the camera feed
            frozenFrame?.let { frame ->
                Image(
                    bitmap = frame,
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            }

            // Viewfinder Scanner Guide Overlay
            ViewfinderOverlay()

            // Single-pass Canvas for bounding boxes inside the 4:3 viewport
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
        }

        // Top Floating Bar (Flash, Pause/Freeze, Zoom, Copy)
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(top = 16.dp)
        ) {
            TopFloatingBar(
                isTorchOn = isTorchOn,
                hasFlash = hasFlash,
                isFrozen = isFrozen,
                zoomRatio = zoomRatio,
                maxZoomRatio = maxZoomRatio,
                onToggleFlash = { isTorchOn = !isTorchOn },
                onToggleFreeze = { toggleFreeze() },
                onToggleZoom = {
                    zoomRatio = if (zoomRatio < 1.8f) {
                        2f.coerceAtMost(maxZoomRatio)
                    } else {
                        1f
                    }
                },
                onCopyAll = { copyText("Recognized text", displayText) },
                hasText = displayText.isNotEmpty()
            )
        }

        // Bottom overlay: Bank details card displayed ONLY when an account number has been identified
        detectedNuban?.let { candidate ->
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(20.dp)
            ) {
                NubanResultCard(
                    candidate = candidate,
                    onCopyAccount = { copyText("NUBAN account number", it) },
                    onVerify = { accountNum ->
                        Toast.makeText(context, "Verifying $accountNum...", Toast.LENGTH_SHORT).show()
                    }
                )
            }
        }
    }
}

@Composable
private fun TopFloatingBar(
    isTorchOn: Boolean,
    hasFlash: Boolean,
    isFrozen: Boolean,
    zoomRatio: Float,
    maxZoomRatio: Float,
    onToggleFlash: () -> Unit,
    onToggleFreeze: () -> Unit,
    onToggleZoom: () -> Unit,
    onCopyAll: () -> Unit,
    hasText: Boolean
) {
    Surface(
        color = Color(0xFF1E1E22).copy(alpha = 0.95f),
        shape = RoundedCornerShape(50.dp),
        tonalElevation = 8.dp,
        modifier = Modifier.wrapContentSize()
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 1. Flash Icon Button (Lightning bolt)
            if (hasFlash) {
                IconButton(onClick = onToggleFlash) {
                    Icon(
                        imageVector = if (isTorchOn) Icons.Filled.FlashOn else Icons.Filled.FlashOff,
                        contentDescription = "Flash",
                        tint = if (isTorchOn) Color(0xFFFFD600) else Color.White
                    )
                }
            }

            // 2. Freeze / Unfreeze White Pill Button (Pause icon + Snowflake icon side-by-side)
            Surface(
                color = Color.White,
                shape = RoundedCornerShape(50.dp),
                modifier = Modifier
                    .clickable(onClick = onToggleFreeze)
                    .padding(horizontal = 14.dp, vertical = 7.dp)
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = if (isFrozen) Icons.Filled.PlayArrow else Icons.Filled.Pause,
                        contentDescription = "Pause",
                        tint = Color.Black,
                        modifier = Modifier.size(18.dp)
                    )
                    Icon(
                        imageVector = Icons.Filled.AcUnit,
                        contentDescription = "Freeze Snowflake",
                        tint = if (isFrozen) Color(0xFF007AFF) else Color.Black,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            // 3. Zoom Pill Button (1.0x)
            if (maxZoomRatio > 1f) {
                Surface(
                    color = Color(0xFF333338),
                    shape = RoundedCornerShape(50.dp),
                    modifier = Modifier
                        .clickable(enabled = !isFrozen, onClick = onToggleZoom)
                        .padding(horizontal = 14.dp, vertical = 8.dp)
                ) {
                    Text(
                        text = String.format(Locale.US, "%.1fx", zoomRatio),
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        color = Color.White
                    )
                }
            }

            // 4. Copy Icon Button (Overlapping pages / ContentCopy)
            IconButton(
                enabled = hasText,
                onClick = onCopyAll
            ) {
                Icon(
                    imageVector = Icons.Filled.ContentCopy,
                    contentDescription = "Copy All Text",
                    tint = if (hasText) Color.White else Color(0xFF71717A)
                )
            }
        }
    }
}

@Composable
private fun ViewfinderOverlay() {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val width = size.width
        val height = size.height
        val boxWidth = width * 0.9f
        val boxHeight = height * 0.85f
        val left = (width - boxWidth) / 2f
        val top = (height - boxHeight) / 2f
        val right = left + boxWidth
        val bottom = top + boxHeight
        val cornerLength = 28.dp.toPx()
        val strokeWidth = 3.5.dp.toPx()
        val color = Color.White.copy(alpha = 0.85f)

        // Top-Left corner
        drawLine(color, androidx.compose.ui.geometry.Offset(left, top), androidx.compose.ui.geometry.Offset(left + cornerLength, top), strokeWidth)
        drawLine(color, androidx.compose.ui.geometry.Offset(left, top), androidx.compose.ui.geometry.Offset(left, top + cornerLength), strokeWidth)

        // Top-Right corner
        drawLine(color, androidx.compose.ui.geometry.Offset(right, top), androidx.compose.ui.geometry.Offset(right - cornerLength, top), strokeWidth)
        drawLine(color, androidx.compose.ui.geometry.Offset(right, top), androidx.compose.ui.geometry.Offset(right, top + cornerLength), strokeWidth)

        // Bottom-Left corner
        drawLine(color, androidx.compose.ui.geometry.Offset(left, bottom), androidx.compose.ui.geometry.Offset(left + cornerLength, bottom), strokeWidth)
        drawLine(color, androidx.compose.ui.geometry.Offset(left, bottom), androidx.compose.ui.geometry.Offset(left, bottom - cornerLength), strokeWidth)

        // Bottom-Right corner
        drawLine(color, androidx.compose.ui.geometry.Offset(right, bottom), androidx.compose.ui.geometry.Offset(right - cornerLength, bottom), strokeWidth)
        drawLine(color, androidx.compose.ui.geometry.Offset(right, bottom), androidx.compose.ui.geometry.Offset(right, bottom - cornerLength), strokeWidth)
    }
}

@Composable
private fun NubanResultCard(
    candidate: NubanCandidate,
    onCopyAccount: (String) -> Unit,
    onVerify: (String) -> Unit
) {
    Surface(
        color = Color(0xFF1C1C1E),
        shape = RoundedCornerShape(32.dp),
        tonalElevation = 12.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(22.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // 1. Header Row: Status Circle Dot + "BANK DETECTED •" + MoreVert Icon
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(
                                color = when (candidate.confidence) {
                                    Confidence.HIGH -> Color(0xFF00E676)
                                    Confidence.MEDIUM -> Color(0xFFFFAB40)
                                    Confidence.LOW -> Color(0xFFFF5252)
                                },
                                shape = RoundedCornerShape(50)
                            )
                    )
                    Text(
                        text = "BANK DETECTED  •",
                        style = MaterialTheme.typography.labelLarge.copy(
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.2.sp
                        ),
                        color = Color(0xFFA1A1AA)
                    )
                }

                IconButton(
                    onClick = { /* More actions for detected bank details */ },
                    modifier = Modifier.size(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.MoreVert,
                        contentDescription = "More Actions",
                        tint = Color(0xFFA1A1AA)
                    )
                }
            }

            // 2. Large Bold Monospace Account Number
            Text(
                text = candidate.number,
                style = MaterialTheme.typography.displayMedium.copy(
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                ),
                color = Color.White
            )

            // 3. Bank Name Row: Building Icon in circular badge + "ZENITH BANK"
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Surface(
                    color = Color(0xFF2C2C32),
                    shape = RoundedCornerShape(50),
                    modifier = Modifier.size(28.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Filled.AccountBalance,
                            contentDescription = "Bank Icon",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
                Text(
                    text = candidate.bank?.let { it.name.uppercase() } ?: "UNKNOWN BANK",
                    style = MaterialTheme.typography.titleMedium.copy(
                        fontWeight = FontWeight.SemiBold,
                        letterSpacing = 0.8.sp
                    ),
                    color = Color(0xFFE4E4E7)
                )
            }

            Spacer(Modifier.height(4.dp))

            // 4. Action Buttons Row: "Copy Number" & "Verify"
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Button(
                    onClick = { onCopyAccount(candidate.number) },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFE4E4E8),
                        contentColor = Color(0xFF111113)
                    ),
                    shape = RoundedCornerShape(50.dp),
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp)
                ) {
                    Text(
                        text = "Copy Number",
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }

                OutlinedButton(
                    onClick = { onVerify(candidate.number) },
                    border = BorderStroke(1.dp, Color(0xFF55555A)),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = Color.White
                    ),
                    shape = RoundedCornerShape(50.dp),
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp)
                ) {
                    Text(
                        text = "Verify",
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }
        }
    }
}