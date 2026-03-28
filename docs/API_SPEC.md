# API Specification — KrishiRakshak v1

## Base URL
- Local: `http://localhost:8000/v1`
- Production: `https://api.krishirakshak.example.com/v1`

---

## Endpoints

### POST /v1/diagnose
Diagnose crop disease from a leaf image.

**Request:**
```
Content-Type: multipart/form-data

Fields:
  image: file (JPEG/PNG, max 5MB) — REQUIRED
  language: string (default: "en-IN") — OPTIONAL
    Supported: en-IN, hi-IN, mr-IN, ta-IN, te-IN, kn-IN, bn-IN
  include_audio: boolean (default: true) — OPTIONAL
```

**Response (200):**
```json
{
  "request_id": "uuid-v4",
  "disease": {
    "name": "Tomato Early Blight",
    "crop": "Tomato",
    "severity": "moderate",
    "confidence": 0.92,
    "confidence_level": "high"
  },
  "treatment": {
    "english": "Remove infected leaves immediately. Apply Mancozeb 75WP...",
    "translated": "संक्रमित पत्तियों को तुरंत हटाएं...",
    "language": "hi-IN"
  },
  "audio": {
    "base64": "UklGRi...",
    "format": "wav",
    "sample_rate": 22050
  },
  "metadata": {
    "model_version": "1.1.0",
    "inference_time_ms": 1250,
    "timestamp": "2026-03-28T10:30:00Z"
  }
}
```

**Response (400):**
```json
{
  "error": "INVALID_IMAGE",
  "message": "Uploaded file is not a valid image. Supported formats: JPEG, PNG.",
  "request_id": "uuid-v4"
}
```

**Response (422) — Low confidence:**
```json
{
  "request_id": "uuid-v4",
  "disease": {
    "name": "uncertain",
    "confidence": 0.32,
    "confidence_level": "low"
  },
  "treatment": {
    "english": "The AI system is not confident enough to provide a diagnosis. Please take a clearer photo or consult your local agricultural extension officer."
  },
  "audio": null
}
```

---

### POST /v1/feedback
Submit feedback on a diagnosis.

**Request:**
```json
{
  "request_id": "uuid-from-diagnose-response",
  "is_correct": false,
  "actual_disease": "Tomato Late Blight",
  "comment": "Leaves had water-soaked lesions, not dry spots"
}
```

**Response (200):**
```json
{
  "status": "recorded",
  "request_id": "uuid-v4",
  "message": "Thank you for your feedback. This helps improve our system."
}
```

---

### GET /v1/health
Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "model_version": "1.1.0",
  "sagemaker_endpoint": "active",
  "sarvam_api": "reachable",
  "uptime_seconds": 3600,
  "timestamp": "2026-03-28T10:30:00Z"
}
```

---

### GET /v1/languages
List supported languages.

**Response (200):**
```json
{
  "languages": [
    {"code": "en-IN", "name": "English", "tts_supported": true},
    {"code": "hi-IN", "name": "Hindi", "tts_supported": true},
    {"code": "mr-IN", "name": "Marathi", "tts_supported": false},
    {"code": "ta-IN", "name": "Tamil", "tts_supported": true},
    {"code": "te-IN", "name": "Telugu", "tts_supported": true},
    {"code": "kn-IN", "name": "Kannada", "tts_supported": true},
    {"code": "bn-IN", "name": "Bengali", "tts_supported": true}
  ]
}
```

---

## Error Codes
| Code | Meaning |
|------|---------|
| INVALID_IMAGE | Not a valid image file or exceeds 5MB |
| UNSUPPORTED_FORMAT | Image format not in JPEG/PNG |
| MODEL_UNAVAILABLE | SageMaker endpoint unreachable |
| TRANSLATION_FAILED | Sarvam API error (response still includes English) |
| TTS_FAILED | Sarvam TTS error (response still includes text) |
| RATE_LIMITED | Too many requests |
| INTERNAL_ERROR | Unexpected server error |

## Headers
- `X-Request-ID`: Unique request identifier (returned in every response)
- `X-Model-Version`: Model version used for inference
- `X-Inference-Time-Ms`: Total inference time in milliseconds
