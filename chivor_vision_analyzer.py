import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

INPUT_DIR = r"C:\Users\lagus\.gemini\antigravity\brain\b038c308-13b8-4caf-9619-4f19368d1620\scratch\piedras"
ARTIFACT_DIR = r"C:\Users\lagus\.gemini\antigravity\brain\b038c308-13b8-4caf-9619-4f19368d1620"
PROJECT_OUT_DIR = r"C:\Users\lagus\.gemini\antigravity\scratch\chivor_emerald_ai\output"

os.makedirs(PROJECT_OUT_DIR, exist_ok=True)


def get_rock_roi_mask(img_bgr, filename):
    """
    Isolates the rock specimen from background (desk, keyboard, hand, bottle cap)
    using spatial ROI polygon + GrabCut/color exclusion of skin, dark mousepad, and blue plastic cap.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # Spatial mask per photo layout (all rocks are in the central region, away from top-left bottle cap and right keyboard)
    spatial = np.zeros((h, w), dtype=np.uint8)
    if "(1)" in filename:
        # Small rock in hand
        pts = np.array([
            [int(w*0.32), int(h*0.47)],
            [int(w*0.64), int(h*0.47)],
            [int(w*0.82), int(h*0.53)],
            [int(w*0.82), int(h*0.65)],
            [int(w*0.72), int(h*0.75)],
            [int(w*0.33), int(h*0.76)],
        ], np.int32)
        cv2.fillPoly(spatial, [pts], 255)
    elif "(2)" in filename:
        # Medium rock held in hand showing pyrite bands
        pts = np.array([
            [int(w*0.14), int(h*0.26)],
            [int(w*0.76), int(h*0.28)],
            [int(w*0.83), int(h*0.73)],
            [int(w*0.48), int(h*0.80)],
            [int(w*0.31), int(h*0.73)],
            [int(w*0.15), int(h*0.53)],
        ], np.int32)
        cv2.fillPoly(spatial, [pts], 255)
    elif "(3)" in filename:
        # Tall profile view of large rock
        pts = np.array([
            [int(w*0.42), int(h*0.18)],
            [int(w*0.70), int(h*0.21)],
            [int(w*0.86), int(h*0.68)],
            [int(w*0.83), int(h*0.88)],
            [int(w*0.52), int(h*0.80)],
            [int(w*0.44), int(h*0.58)],
        ], np.int32)
        cv2.fillPoly(spatial, [pts], 255)
    elif "(4)" in filename:
        # Carpet of emeralds view
        pts = np.array([
            [int(w*0.19), int(h*0.31)],
            [int(w*0.65), int(h*0.31)],
            [int(w*0.91), int(h*0.46)],
            [int(w*0.88), int(h*0.60)],
            [int(w*0.42), int(h*0.67)],
            [int(w*0.23), int(h*0.56)],
        ], np.int32)
        cv2.fillPoly(spatial, [pts], 255)
    else:
        # Main block flat on desk
        pts = np.array([
            [int(w*0.21), int(h*0.14)],
            [int(w*0.75), int(h*0.14)],
            [int(w*0.86), int(h*0.50)],
            [int(w*0.82), int(h*0.71)],
            [int(w*0.68), int(h*0.84)],
            [int(w*0.33), int(h*0.87)],
            [int(w*0.26), int(h*0.70)],
        ], np.int32)
        cv2.fillPoly(spatial, [pts], 255)

    # Exclude skin tones inside the polygon (e.g., thumb in photo (3))
    # Skin in YCrCb has high Cr (138..178) and moderate Cb (95..128) while rock is neutral or green
    skin_mask = cv2.inRange(ycrcb, (60, 140, 95), (255, 180, 130))
    # Ensure we never exclude green pixels
    green_protect = cv2.inRange(hsv, (35, 30, 30), (98, 255, 255))
    skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(green_protect))

    # Exclude dark black mousepad inside polygon
    dark_pad = cv2.inRange(hsv, (0, 0, 0), (180, 90, 65))

    rock_mask = cv2.bitwise_and(spatial, cv2.bitwise_not(skin_mask))
    rock_mask = cv2.bitwise_and(rock_mask, cv2.bitwise_not(dark_pad))

    # Smooth morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    rock_mask = cv2.morphologyEx(rock_mask, cv2.MORPH_CLOSE, kernel)
    return rock_mask


def analyze_rock_specimen(img_bgr, filename):
    """
    Performs multi-mineral segmentation:
    1. Emerald (Cr3+/V bluish-green beryl + simulated 680nm LIF intensity)
    2. Albite/Carbonate hydrothermal vein ("Cama" blanca)
    3. Pyrite (FeS2 golden sulfide trails)
    4. Host Albitized Siltstone ("Roca Caja")
    """
    h, w = img_bgr.shape[:2]
    rock_mask = get_rock_roi_mask(img_bgr, filename)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(img_bgr.astype(np.float32))

    # 1. EMERALD DETECTION (Chivor bluish-green signature: Hue 42..95 in OpenCV HSV + excess Green-Blue over Red)
    emerald_hsv = cv2.inRange(hsv, (42, 38, 45), (96, 255, 255))
    # Chromatic index: Chivor emeralds have strong G > R and B >= R*0.85
    emerald_chrom = ((g - r > 14) & (g > b * 0.92)).astype(np.uint8) * 255
    emerald_mask = cv2.bitwise_and(emerald_hsv, emerald_chrom)
    emerald_mask = cv2.bitwise_and(emerald_mask, rock_mask)

    # Clean noise and consolidate crystal clusters
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    emerald_mask = cv2.morphologyEx(emerald_mask, cv2.MORPH_OPEN, k_small)
    emerald_mask = cv2.morphologyEx(emerald_mask, cv2.MORPH_CLOSE, k_med)

    # 2. ALBITE / CARBONATE VEIN ("Cama blanca hidrotermal")
    # High brightness (L > 188 in LAB or V > 195), low saturation (S < 38)
    albite_mask = cv2.inRange(hsv, (0, 0, 192), (180, 42, 255))
    albite_mask = cv2.bitwise_and(albite_mask, rock_mask)
    albite_mask = cv2.bitwise_and(albite_mask, cv2.bitwise_not(emerald_mask))
    albite_mask = cv2.morphologyEx(albite_mask, cv2.MORPH_OPEN, k_small)

    # 3. PYRITE BANDS / SULFIDE MINERALIZATION (FeS2 micro-crystals & golden-bronze streaks)
    # Pyrite in matrix shows localized high-contrast golden-olive/bronze speckles (Hue 14..34, S 45..180, high local gradient)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    high_tex = (laplacian > 18).astype(np.uint8) * 255
    high_tex = cv2.dilate(high_tex, k_small, iterations=1)

    pyrite_hsv = cv2.inRange(hsv, (14, 42, 55), (33, 210, 215))
    pyrite_mask = cv2.bitwise_and(pyrite_hsv, high_tex)
    pyrite_mask = cv2.bitwise_and(pyrite_mask, rock_mask)
    pyrite_mask = cv2.bitwise_and(pyrite_mask, cv2.bitwise_not(emerald_mask))
    pyrite_mask = cv2.bitwise_and(pyrite_mask, cv2.bitwise_not(albite_mask))

    # Simulated 680nm Laser-Induced Fluorescence (LIF) response map:
    # Proportional to Cr3+ chromatic excess (G - R) inside the emerald mask with optical glow halo
    cr_excess = np.clip((g - r) * 3.2, 0, 255).astype(np.uint8)
    lif_680 = cv2.bitwise_and(cr_excess, cr_excess, mask=emerald_mask)
    lif_glow = cv2.GaussianBlur(lif_680, (25, 25), 0)
    lif_680 = np.clip(lif_680.astype(np.float32) + lif_glow.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)

    # Build Stingray-style HUD visualization
    # Darkened grayscale rock background (like the Stingray machine vision camera)
    hud = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hud = (hud.astype(np.float32) * 0.45).astype(np.uint8)
    # Slightly tint the active rock zone
    rock_tint = np.zeros_like(hud)
    rock_tint[:, :] = (35, 30, 25)
    hud = np.where(rock_mask[:, :, None] > 0, cv2.add(hud, rock_tint), (hud * 0.5).astype(np.uint8))

    # Overlay Albite vein in Cyan-Blue tint
    hud[albite_mask > 0] = (
        hud[albite_mask > 0].astype(np.float32) * 0.35 + np.array([230, 190, 70], dtype=np.float32) * 0.65
    ).astype(np.uint8)

    # Overlay Pyrite in Golden-Yellow
    hud[pyrite_mask > 0] = (0, 215, 255)

    # Overlay Emerald in Vivid Neon Green
    hud[emerald_mask > 0] = (60, 255, 0)

    # Draw rock boundary contour in subtle tech cyan
    rock_cnts, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hud, rock_cnts, -1, (180, 140, 60), 2)

    # Find individual emerald crystal targets & draw Stingray-style bounding boxes + laser target reticles
    cnts, _ = cv2.findContours(emerald_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 35:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + cw // 2, y + ch // 2
        targets.append((area, x, y, cw, ch, cx, cy, c))

    targets.sort(key=lambda t: t[0], reverse=True)

    # Draw Laser Beam Vectors from top-center scanner node to top emerald targets (Stingray style!)
    scanner_origin = (w // 2, 30)
    for idx, (area, x, y, cw, ch, cx, cy, c) in enumerate(targets[:12]):
        # Bounding box
        cv2.rectangle(hud, (x - 4, y - 4), (x + cw + 4, y + ch + 4), (0, 255, 120), 2)
        # Crosshair reticle on crystal centroid
        cv2.circle(hud, (cx, cy), 6, (0, 0, 255), 2)
        cv2.circle(hud, (cx, cy), 2, (255, 255, 255), -1)
        if idx < 5:
            # Draw laser lock line
            cv2.line(hud, scanner_origin, (cx, cy), (80, 255, 80), 1, cv2.LINE_AA)
            label = f"CR3+ #{idx+1} [{int(area)}px]"
            cv2.putText(
                hud, label, (max(10, x - 5), max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 140), 1, cv2.LINE_AA
            )

    # Fit structural vein line if enough emerald/albite pixels exist
    vein_pts = np.column_stack(np.where(emerald_mask > 0))
    if len(vein_pts) > 50:
        # y, x coordinates -> convert to x, y for fitLine
        xy_pts = np.column_stack((vein_pts[:, 1], vein_pts[:, 0])).astype(np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(xy_pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
        lefty = int((-x0 * vy / (vx + 1e-6)) + y0)
        righty = int(((w - x0) * vy / (vx + 1e-6)) + y0)
        # Draw structural vein axis in dashed/bright magenta-red
        cv2.line(hud, (0, lefty), (w - 1, righty), (255, 80, 255), 2, cv2.LINE_AA)

    # Compute metrics (%)
    rock_px = max(1, np.count_nonzero(rock_mask))
    emerald_pct = (np.count_nonzero(emerald_mask) / rock_px) * 100.0
    albite_pct = (np.count_nonzero(albite_mask) / rock_px) * 100.0
    pyrite_pct = (np.count_nonzero(pyrite_mask) / rock_px) * 100.0

    # HUD Header overlay
    cv2.rectangle(hud, (0, 0), (w, 95), (18, 18, 18), -1)
    cv2.putText(hud, "CHIVOR AI-LASER VISION HUD // LIF 680nm + SPECTRAL MINERALOGY", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 160), 2, cv2.LINE_AA)
    cv2.putText(hud, f"ESMERALDA (Cr3+): {emerald_pct:.2f}% | CAMA ALBITA: {albite_pct:.1f}% | PIRITA (FeS2): {pyrite_pct:.2f}%",
                (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(hud, f"TARGETS DETECTADOS: {len(targets)} | EJE DE VETA (MAGENTA) | ESTADO: BLOQUEAR EXPLOSIVO",
                (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 190, 255), 1, cv2.LINE_AA)

    return hud, lif_680, emerald_mask, albite_mask, pyrite_mask, (emerald_pct, albite_pct, pyrite_pct, len(targets))


def main():
    files = [
        ("WhatsApp Image 2026-11-10 at 9.35.26 PM.jpeg", "1_bloque_veta_perfil", "Muestra 1 (Perfil): Veta milimetrica (4mm) en Roca Gris Albitizada"),
        ("WhatsApp Image 2026-11-10 at 9.35.26 PM (4).jpeg", "2_bloque_alfombra_esmeralda", "Muestra 1 (Abierta): Alfombra de Esmeraldas sobre Cama de Albita"),
        ("WhatsApp Image 2026-11-10 at 9.35.26 PM (1).jpeg", "3_canutillo_matriz", "Muestra 2 (Frente): Canutillo de Esmeralda en Albita y Pirita"),
        ("WhatsApp Image 2026-11-10 at 9.35.26 PM (2).jpeg", "4_cordones_pirita", "Muestra 2 (Reverso): Cordones Guia de Pirita (FeS2) + Cama Blanca"),
        ("WhatsApp Image 2026-11-10 at 9.35.26 PM (3).jpeg", "5_costura_lateral", "Muestra 1 (Lateral): Costura Hidrotermal para Deteccion con Sonda"),
    ]

    summary_rows = []
    for fname, slug, title in files:
        fpath = os.path.join(INPUT_DIR, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print(f"Error reading {fpath}")
            continue

        hud, lif_680, em_mask, alb_mask, pyr_mask, stats = analyze_rock_specimen(img_bgr, fname)
        emerald_pct, albite_pct, pyrite_pct, n_targets = stats
        summary_rows.append((slug, title, emerald_pct, albite_pct, pyrite_pct, n_targets))

        # Create a 3-panel comparison figure:
        # Panel 1: Original Photo
        # Panel 2: Simulated 680nm Laser Fluorescence (LIF Cr3+ Channel - What the 680nm filter sees in the dark!)
        # Panel 3: Stingray-Style AI Mineralogical HUD + Vein Plane Axis
        fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), dpi=140, facecolor="#101418")
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.97)

        # Panel 1: Original
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Vista Humana (Luz Blanca)", color="#cfd8dc", fontsize=11, pad=8)
        axes[0].axis("off")

        # Panel 2: 680nm LIF Channel (Dark background, glowing crimson-red Cr3+ luminescence)
        lif_rgb = np.zeros_like(img_bgr)
        # Map LIF intensity to glowing red-orange-white (680nm R-line appearance)
        lif_norm = lif_680.astype(np.float32) / 255.0
        lif_rgb[:, :, 0] = np.clip(lif_norm * 255 * 1.3, 0, 255).astype(np.uint8)  # Red
        lif_rgb[:, :, 1] = np.clip((lif_norm ** 2) * 180, 0, 255).astype(np.uint8)  # Hot core orange/yellow
        lif_rgb[:, :, 2] = np.clip((lif_norm ** 3) * 220, 0, 255).astype(np.uint8)  # White peak
        # Add subtle rock outline so viewer sees where rock is in complete darkness
        rock_mask = get_rock_roi_mask(img_bgr, fname)
        cnts, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(lif_rgb, cnts, -1, (45, 55, 65), 2)

        axes[1].imshow(lif_rgb)
        axes[1].set_title("2. Canal Laser 680nm (Fluorescencia Cr3+ en Oscuridad)", color="#ff5252", fontsize=11, pad=8)
        axes[1].axis("off")

        # Panel 3: AI HUD
        axes[2].imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))
        axes[2].set_title("3. HUD Robot IA (Verde=Esmeralda | Azul=Albita | Oro=Pirita)", color="#00e676", fontsize=11, pad=8)
        axes[2].axis("off")

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])

        out_artifact = os.path.join(ARTIFACT_DIR, f"hud_{slug}.png")
        out_project = os.path.join(PROJECT_OUT_DIR, f"hud_{slug}.png")
        fig.savefig(out_artifact, facecolor=fig.get_facecolor(), bbox_inches="tight")
        fig.savefig(out_project, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_artifact} | Emerald={emerald_pct:.2f}%, Albite={albite_pct:.1f}%, Pyrite={pyrite_pct:.2f}%, Targets={n_targets}")


if __name__ == "__main__":
    main()
