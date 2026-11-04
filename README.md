# 🛰️ Satellite Image Segmentation: Web Frontend

> **A React single-page app for the Satellite Intelligence & Geospatial Computer Vision project. Upload an aerial or satellite image and see the original next to the model's building-segmentation overlay.**

This is the `frontend` branch. The model, training code, evaluation, and the FastAPI service live on the [`main` branch](https://github.com/ishaan175pathak/Satellite-Intelligence-Geospatial-Computer-Vision/tree/main), which also holds the full project write-up, results, and API reference.

---

## What It Does

1. You choose an image (JPEG or PNG).
2. The app sends it to the backend with `POST /uploads`.
3. The backend runs the U-Net + SegFormer-B0 ensemble and returns the image with the predicted building regions overlaid in red, as a base64-encoded PNG.
4. The page shows the original image and the segmentation overlay side by side, with loading and error states.

The ensemble reaches a test Dice of 0.9420 and an object-level F1 of 0.8600 on the WHU Building Dataset. See the `main` README for the full evaluation and its caveats.

```text
 Browser (this app)
      │  POST /uploads (multipart image)
      ▼
 FastAPI backend (main branch) ── U-Net + SegFormer-B0 ensemble
      │  JSON with base64 PNG overlay
      ▼
 Browser: original image | segmentation overlay
```

---

## Tech Stack

* React 19
* Vite
* Tailwind CSS
* ESLint

---

## Getting Started

### Prerequisites

* A current LTS version of Node.js and npm
* The backend running at `http://localhost:8000` (see the [`main` README](https://github.com/ishaan175pathak/Satellite-Intelligence-Geospatial-Computer-Vision/tree/main#-deployment), section *Deployment*)

### Run the App

```bash
npm install
npm run dev
```

Open the address that Vite prints (by default `http://localhost:5173`) and choose an image.

### Other Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Start the development server |
| `npm run build` | Create a production build in `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run ESLint |

### Start the Backend

The backend lives on the `main` branch, so use a second checkout (for example `git worktree add ../satellite-backend main`) and run, from its root:

```bash
uvicorn src.api.app:app --reload
```

The backend needs the trained model checkpoints in its root directory. The `main` README explains how to produce them.

---

## Configuration

The API address is set directly in `src/components/UserInput.jsx`:

```js
fetch('http://localhost:8000/uploads', { method: 'POST', body: formData })
```

To use a different host or port, change that URL.

The browser sends this request from the Vite origin (`http://localhost:5173`) to the API origin (`http://localhost:8000`), so the backend must allow cross-origin requests from the frontend's origin (CORS). Without that, the browser blocks the response and the app shows "An error occurred while processing the image."

---

## Project Structure

```text
.
├── index.html
├── package.json
├── vite.config.js
├── eslint.config.js
├── public/
└── src/
    ├── main.jsx                 # App entry point
    ├── App.jsx
    ├── layouts/
    │   └── MainLayout.jsx       # Navbar + upload view
    ├── components/
    │   ├── Navbar.jsx
    │   ├── UserInput.jsx        # Upload form, API call, result display
    │   └── MapComponent.jsx     # Empty placeholder, not used
    ├── styles/
    │   └── mainLayout.css
    ├── App.css
    └── index.css                # Tailwind import
```

---

## Limitations

* One image at a time; the API accepts JPEG and PNG only and rejects other formats.
* The backend serves the ensemble only, so each request runs both networks. Latency was not benchmarked.
* The overlay is an image. The app does not show polygons, GeoJSON, or map coordinates.
* The models were trained on aerial imagery from one region (Christchurch, NZ). Results on other regions, sensors, or ground resolutions were not evaluated.

---

## Author

**Ishaan Pathak**, MS in Computer Science