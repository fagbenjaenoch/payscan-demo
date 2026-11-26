package com.example.payscandemo

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * The only file that knows about LiteRT. Needs in app/build.gradle.kts:
 *     implementation("com.google.ai.edge.litert:litert:2.3.0")
 * and the model at app/src/main/assets/nuban_digits.tflite
 *
 * Create it, call predict() and close() on the SAME thread (the analyzer's executor).
 * If the LiteRT calls below do not compile, compare them with the CompiledModel Kotlin sample at
 * https://developers.google.com/edge/litert/android; the rest of the app only uses [DigitModel].
 */
class LiteRtDigitModel(
    context: Context,
    assetName: String = "nuban_digits.tflite",
) : DigitModel, AutoCloseable {

    private val model = CompiledModel.create(context.assets, assetName, CompiledModel.Options(Accelerator.CPU))
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()

    override fun predict(crops: List<FloatArray>): List<FloatArray> =
        crops.map { crop ->
            inputs[0].writeFloat(crop)            // 32*32 floats, 0..1, ink = 1
            model.run(inputs, outputs)
            outputs[0].readFloat()                // 11 probabilities: digits 0-9, then "not a digit"
        }

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
    }
}
