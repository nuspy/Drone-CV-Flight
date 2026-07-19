package com.dronecv.photolocator

/** A geographic fix computed from one photo (either mode). */
data class LocalizationResult(
    val lat: Double,
    val lon: Double,
    val altMsl: Double,
    val headingDeg: Double,
    val confidence: Double,
    val sigmaHM: Double,
    val source: String, // "server" | "on-device"
    val inferenceMs: Long,
)
