"""
Application GUI – Segmentation d'images en présence de bruit

Interface à 4 onglets :
  1. Bruit        – charger image, ajouter bruit Gaussien / S&P, voir PSNR
  2. Débruitage   – appliquer Gaussien / Médian / Bilatéral / NLM
  3. Segmentation – Canny / Otsu / Adaptatif / Watershed + pipeline robuste
  4. Évaluation   – benchmark multi-bruit, métriques IoU/Dice/F1, graphes
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── projet local ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from noise import add_noise, compute_psnr, compute_snr
from denoising import denoise
from segmentation.classical import segment_classical, watershed_segmentation
from evaluation import evaluate_mask, benchmark_noise_levels

# ─────────────────────────────────────────────────────────────────────────────
# Palette & constantes
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
BG2       = "#2a2a3e"
ACCENT    = "#7c6af7"
ACCENT2   = "#a78bfa"
FG        = "#e2e8f0"
FG2       = "#94a3b8"
BTN_BG    = "#3b3b5c"
BTN_HOV   = "#4c4c6d"
SUCCESS   = "#22c55e"
WARNING   = "#f59e0b"
DANGER    = "#ef4444"
FONT      = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_BIG  = ("Segoe UI", 14, "bold")
FONT_SM   = ("Segoe UI", 9)

PREVIEW_W = 320
PREVIEW_H = 280


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires image → PhotoImage
# ─────────────────────────────────────────────────────────────────────────────

def cv2_to_photoimage(img_bgr, max_w=PREVIEW_W, max_h=PREVIEW_H):
    if img_bgr is None:
        return None
    if img_bgr.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img_resized))


def placeholder_photoimage(text="Aucune image", w=PREVIEW_W, h=PREVIEW_H):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(img, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 120), 1, cv2.LINE_AA)
    return ImageTk.PhotoImage(Image.fromarray(img))


# ─────────────────────────────────────────────────────────────────────────────
# Widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def styled_btn(parent, text, command, color=BTN_BG, width=18):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=color, fg=FG, activebackground=BTN_HOV, activeforeground=FG,
        relief="flat", bd=0, padx=10, pady=6,
        font=FONT_BOLD, cursor="hand2", width=width,
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HOV))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


def label_val(parent, label, row, col=0, val="—"):
    tk.Label(parent, text=label, bg=BG2, fg=FG2, font=FONT_SM).grid(
        row=row, column=col, sticky="w", padx=6, pady=2)
    var = tk.StringVar(value=val)
    tk.Label(parent, textvariable=var, bg=BG2, fg=ACCENT2, font=FONT_BOLD).grid(
        row=row, column=col + 1, sticky="w", padx=6, pady=2)
    return var


def section_label(parent, text):
    tk.Label(parent, text=text, bg=BG, fg=ACCENT2, font=FONT_BIG).pack(anchor="w", padx=16, pady=(14, 4))


def separator(parent):
    ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=16, pady=4)


def image_panel(parent, title, w=PREVIEW_W, h=PREVIEW_H):
    frame = tk.Frame(parent, bg=BG2, bd=0)
    tk.Label(frame, text=title, bg=BG2, fg=FG2, font=FONT_SM).pack(pady=(6, 2))
    lbl = tk.Label(frame, bg=BG2, width=w, height=h)
    lbl.pack(padx=8, pady=(0, 8))
    ph = placeholder_photoimage(title, w, h)
    lbl.config(image=ph)
    lbl._photo = ph
    return frame, lbl


def update_image_label(lbl, img_bgr):
    ph = cv2_to_photoimage(img_bgr)
    if ph:
        lbl.config(image=ph, width=ph.width(), height=ph.height())
        lbl._photo = ph


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 1 – Bruit
# ─────────────────────────────────────────────────────────────────────────────

class TabNoise(tk.Frame):
    def __init__(self, master, shared):
        super().__init__(master, bg=BG)
        self.shared = shared

        # ── Controls (gauche) ──────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, width=260)
        ctrl.pack(side="left", fill="y", padx=(16, 8), pady=16)
        ctrl.pack_propagate(False)

        section_label(ctrl, "1. Chargement")
        styled_btn(ctrl, "Ouvrir une image", self._load_image, ACCENT).pack(anchor="w", padx=16, pady=4)

        separator(ctrl)
        section_label(ctrl, "2. Type de bruit")

        self.noise_type = tk.StringVar(value="gaussian")
        for txt, val in [("Gaussien", "gaussian"), ("Sel-et-Poivre", "salt_and_pepper")]:
            tk.Radiobutton(ctrl, text=txt, variable=self.noise_type, value=val,
                           bg=BG, fg=FG, selectcolor=ACCENT, activebackground=BG,
                           font=FONT, command=self._update_slider_label).pack(anchor="w", padx=16)

        separator(ctrl)
        section_label(ctrl, "3. Niveau de bruit")

        self.slider_label = tk.Label(ctrl, text="Écart-type σ", bg=BG, fg=FG2, font=FONT_SM)
        self.slider_label.pack(anchor="w", padx=16)

        self.noise_level = tk.DoubleVar(value=25)
        self.slider = tk.Scale(ctrl, from_=0, to=150, resolution=1,
                               variable=self.noise_level, orient="horizontal",
                               bg=BG, fg=FG, troughcolor=BG2, highlightthickness=0,
                               length=210, sliderlength=18, width=14)
        self.slider.pack(anchor="w", padx=16)

        separator(ctrl)
        styled_btn(ctrl, "Appliquer le bruit", self._apply_noise, ACCENT).pack(anchor="w", padx=16, pady=6)
        styled_btn(ctrl, "Enregistrer image bruitée", self._save_noisy, BTN_BG).pack(anchor="w", padx=16, pady=2)

        separator(ctrl)
        section_label(ctrl, "Métriques")
        info = tk.Frame(ctrl, bg=BG2, bd=0)
        info.pack(fill="x", padx=16, pady=4)
        self.var_psnr = label_val(info, "PSNR :", 0)
        self.var_snr  = label_val(info, "SNR  :", 1)
        self.var_size = label_val(info, "Taille:", 2)

        # ── Panneaux image ─────────────────────────────────────────────────
        panels = tk.Frame(self, bg=BG)
        panels.pack(side="left", fill="both", expand=True, padx=8, pady=16)

        row0 = tk.Frame(panels, bg=BG)
        row0.pack(fill="both", expand=True)

        pf_orig, self.lbl_orig = image_panel(row0, "Image originale")
        pf_orig.pack(side="left", padx=10, pady=6, fill="both", expand=True)

        pf_noisy, self.lbl_noisy = image_panel(row0, "Image bruitée")
        pf_noisy.pack(side="left", padx=10, pady=6, fill="both", expand=True)

        # Histogramme
        self.fig_hist, self.ax_hist = plt.subplots(1, 2, figsize=(7, 2.5))
        self.fig_hist.patch.set_facecolor("#1e1e2e")
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=panels)
        self.canvas_hist.get_tk_widget().pack(fill="x", padx=10, pady=4)

    def _update_slider_label(self):
        if self.noise_type.get() == "gaussian":
            self.slider_label.config(text="Écart-type σ  (0 – 150)")
            self.slider.config(from_=0, to=150, resolution=1)
            self.noise_level.set(25)
        else:
            self.slider_label.config(text="Densité d  (0.00 – 0.50)")
            self.slider.config(from_=0, to=50, resolution=1)
            self.noise_level.set(5)

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Ouvrir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tous", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de lire l'image.")
            return
        self.shared["original"] = img
        self.shared["noisy"]    = None
        update_image_label(self.lbl_orig, img)
        ph = placeholder_photoimage("Image bruitée")
        self.lbl_noisy.config(image=ph)
        self.lbl_noisy._photo = ph
        h, w = img.shape[:2]
        self.var_size.set(f"{w}×{h}")
        self.var_psnr.set("—")
        self.var_snr.set("—")
        self._draw_histograms(img, None)

    def _apply_noise(self):
        if self.shared.get("original") is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image.")
            return
        ntype = self.noise_type.get()
        lvl   = self.noise_level.get()
        orig  = self.shared["original"]

        if ntype == "gaussian":
            noisy = add_noise(orig, "gaussian", sigma=float(lvl))
        else:
            density = lvl / 100.0
            noisy = add_noise(orig, "salt_and_pepper", density=density)

        self.shared["noisy"] = noisy
        psnr = compute_psnr(orig, noisy)
        snr  = compute_snr(orig, noisy)
        self.var_psnr.set(f"{psnr:.2f} dB")
        self.var_snr.set(f"{snr:.2f} dB")
        update_image_label(self.lbl_noisy, noisy)
        self._draw_histograms(orig, noisy)

    def _save_noisy(self):
        if self.shared.get("noisy") is None:
            messagebox.showwarning("Attention", "Aucune image bruitée à sauvegarder.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            cv2.imwrite(path, self.shared["noisy"])
            messagebox.showinfo("Sauvegardé", f"Image enregistrée :\n{path}")

    def _draw_histograms(self, orig, noisy):
        for ax in self.ax_hist:
            ax.clear()
            ax.set_facecolor("#2a2a3e")
            for sp in ax.spines.values():
                sp.set_color("#3b3b5c")
            ax.tick_params(colors=FG2, labelsize=7)

        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim == 3 else orig
        self.ax_hist[0].hist(gray_orig.ravel(), bins=128, range=(0, 255),
                              color=ACCENT, alpha=0.8, linewidth=0)
        self.ax_hist[0].set_title("Originale", color=FG2, fontsize=8)
        self.ax_hist[0].set_xlabel("Valeur pixel", color=FG2, fontsize=7)

        if noisy is not None:
            gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY) if noisy.ndim == 3 else noisy
            self.ax_hist[1].hist(gray_noisy.ravel(), bins=128, range=(0, 255),
                                  color=WARNING, alpha=0.8, linewidth=0)
            self.ax_hist[1].set_title("Bruitée", color=FG2, fontsize=8)
            self.ax_hist[1].set_xlabel("Valeur pixel", color=FG2, fontsize=7)

        self.fig_hist.tight_layout(pad=0.5)
        self.canvas_hist.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 2 – Débruitage
# ─────────────────────────────────────────────────────────────────────────────

class TabDenoise(tk.Frame):
    def __init__(self, master, shared):
        super().__init__(master, bg=BG)
        self.shared   = shared
        self.denoised = None

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, width=260)
        ctrl.pack(side="left", fill="y", padx=(16, 8), pady=16)
        ctrl.pack_propagate(False)

        section_label(ctrl, "Filtre de débruitage")

        self.method = tk.StringVar(value="median")
        filters = [
            ("Gaussien",   "gaussian"),
            ("Médian",     "median"),
            ("Bilatéral",  "bilateral"),
            ("Non-Local Means", "nlm"),
        ]
        for txt, val in filters:
            tk.Radiobutton(ctrl, text=txt, variable=self.method, value=val,
                           bg=BG, fg=FG, selectcolor=ACCENT, activebackground=BG,
                           font=FONT).pack(anchor="w", padx=16, pady=2)

        separator(ctrl)
        section_label(ctrl, "Paramètre du filtre")
        tk.Label(ctrl, text="Taille du noyau (k)", bg=BG, fg=FG2, font=FONT_SM).pack(anchor="w", padx=16)
        self.kernel = tk.IntVar(value=5)
        tk.Scale(ctrl, from_=3, to=21, resolution=2,
                 variable=self.kernel, orient="horizontal",
                 bg=BG, fg=FG, troughcolor=BG2, highlightthickness=0,
                 length=210, sliderlength=18, width=14).pack(anchor="w", padx=16)

        separator(ctrl)
        styled_btn(ctrl, "Appliquer le filtre", self._apply, ACCENT).pack(anchor="w", padx=16, pady=6)
        styled_btn(ctrl, "Enregistrer résultat", self._save, BTN_BG).pack(anchor="w", padx=16, pady=2)

        separator(ctrl)
        section_label(ctrl, "Métriques")
        info = tk.Frame(ctrl, bg=BG2)
        info.pack(fill="x", padx=16, pady=4)
        self.var_psnr_before = label_val(info, "PSNR avant :", 0)
        self.var_psnr_after  = label_val(info, "PSNR après :", 1)
        self.var_gain        = label_val(info, "Gain       :", 2)

        # ── Panneaux ──────────────────────────────────────────────────────
        panels = tk.Frame(self, bg=BG)
        panels.pack(side="left", fill="both", expand=True, padx=8, pady=16)

        row0 = tk.Frame(panels, bg=BG)
        row0.pack(fill="both", expand=True)

        pf1, self.lbl_noisy    = image_panel(row0, "Image bruitée (entrée)")
        pf1.pack(side="left", padx=10, pady=6, fill="both", expand=True)
        pf2, self.lbl_denoised = image_panel(row0, "Après débruitage")
        pf2.pack(side="left", padx=10, pady=6, fill="both", expand=True)

    def _apply(self):
        src = self.shared.get("noisy") or self.shared.get("original")
        if src is None:
            messagebox.showwarning("Attention", "Chargez et/ou bruitez une image d'abord.")
            return
        method = self.method.get()
        k      = self.kernel.get()

        kw = {}
        if method == "gaussian":
            kw = {"kernel_size": k}
        elif method == "median":
            kw = {"kernel_size": k}
        elif method == "bilateral":
            kw = {"d": k}
        # nlm uses defaults

        self.denoised = denoise(src, method, **kw)
        self.shared["denoised"] = self.denoised
        update_image_label(self.lbl_noisy,    src)
        update_image_label(self.lbl_denoised, self.denoised)

        orig = self.shared.get("original")
        if orig is not None:
            psnr_b = compute_psnr(orig, src)
            psnr_a = compute_psnr(orig, self.denoised)
            self.var_psnr_before.set(f"{psnr_b:.2f} dB")
            self.var_psnr_after.set(f"{psnr_a:.2f} dB")
            self.var_gain.set(f"+{psnr_a - psnr_b:.2f} dB")

    def _save(self):
        if self.denoised is None:
            messagebox.showwarning("Attention", "Aucun résultat à sauvegarder.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.denoised)
            messagebox.showinfo("Sauvegardé", f"Image enregistrée :\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 3 – Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class TabSegmentation(tk.Frame):
    def __init__(self, master, shared):
        super().__init__(master, bg=BG)
        self.shared = shared
        self.result = None

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, width=260)
        ctrl.pack(side="left", fill="y", padx=(16, 8), pady=16)
        ctrl.pack_propagate(False)

        section_label(ctrl, "Méthode de segmentation")
        self.method = tk.StringVar(value="otsu")
        methods = [
            ("Canny (contours)",    "canny"),
            ("Otsu (seuillage)",    "otsu"),
            ("Adaptatif (local)",   "adaptive"),
            ("Watershed (régions)", "watershed"),
        ]
        for txt, val in methods:
            tk.Radiobutton(ctrl, text=txt, variable=self.method, value=val,
                           bg=BG, fg=FG, selectcolor=ACCENT, activebackground=BG,
                           font=FONT).pack(anchor="w", padx=16, pady=2)

        separator(ctrl)
        section_label(ctrl, "Source d'entrée")
        self.source = tk.StringVar(value="denoised")
        for txt, val in [("Image débruitée", "denoised"), ("Image bruitée", "noisy"), ("Originale", "original")]:
            tk.Radiobutton(ctrl, text=txt, variable=self.source, value=val,
                           bg=BG, fg=FG, selectcolor=ACCENT, activebackground=BG,
                           font=FONT).pack(anchor="w", padx=16, pady=2)

        separator(ctrl)
        section_label(ctrl, "Paramètres Canny")
        tk.Label(ctrl, text="Seuil bas", bg=BG, fg=FG2, font=FONT_SM).pack(anchor="w", padx=16)
        self.canny_low = tk.IntVar(value=50)
        tk.Scale(ctrl, from_=0, to=200, variable=self.canny_low, orient="horizontal",
                 bg=BG, fg=FG, troughcolor=BG2, highlightthickness=0,
                 length=210, sliderlength=18, width=12).pack(anchor="w", padx=16)
        tk.Label(ctrl, text="Seuil haut", bg=BG, fg=FG2, font=FONT_SM).pack(anchor="w", padx=16)
        self.canny_high = tk.IntVar(value=150)
        tk.Scale(ctrl, from_=0, to=400, variable=self.canny_high, orient="horizontal",
                 bg=BG, fg=FG, troughcolor=BG2, highlightthickness=0,
                 length=210, sliderlength=18, width=12).pack(anchor="w", padx=16)

        separator(ctrl)
        styled_btn(ctrl, "Segmenter", self._apply, ACCENT).pack(anchor="w", padx=16, pady=6)
        styled_btn(ctrl, "Enregistrer masque", self._save, BTN_BG).pack(anchor="w", padx=16, pady=2)

        # ── Panneaux ──────────────────────────────────────────────────────
        panels = tk.Frame(self, bg=BG)
        panels.pack(side="left", fill="both", expand=True, padx=8, pady=16)

        row0 = tk.Frame(panels, bg=BG)
        row0.pack(fill="both", expand=True)

        pf1, self.lbl_input  = image_panel(row0, "Image d'entrée")
        pf1.pack(side="left", padx=10, pady=6, fill="both", expand=True)
        pf2, self.lbl_result = image_panel(row0, "Résultat segmentation")
        pf2.pack(side="left", padx=10, pady=6, fill="both", expand=True)

        # overlay
        pf3, self.lbl_overlay = image_panel(panels, "Overlay (contours sur image)", PREVIEW_W * 2 + 20, PREVIEW_H)
        pf3.pack(fill="x", padx=10, pady=4)

    def _get_source(self):
        key = self.source.get()
        img = self.shared.get(key)
        if img is None:
            for fallback in ("denoised", "noisy", "original"):
                img = self.shared.get(fallback)
                if img is not None:
                    break
        return img

    def _apply(self):
        src = self._get_source()
        if src is None:
            messagebox.showwarning("Attention", "Aucune image disponible.")
            return
        method = self.method.get()

        if method == "canny":
            result = segment_classical(src, "canny",
                                       low_threshold=self.canny_low.get(),
                                       high_threshold=self.canny_high.get())
        elif method == "watershed":
            _, result = watershed_segmentation(src)
        else:
            result = segment_classical(src, method)

        self.result = result
        self.shared["mask"] = result
        update_image_label(self.lbl_input,  src)
        update_image_label(self.lbl_result, result)
        overlay = self._make_overlay(src, result)
        update_image_label(self.lbl_overlay, overlay)

    def _make_overlay(self, src, mask):
        if src.ndim == 2:
            base = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        else:
            base = src.copy()
        if mask.ndim == 2:
            colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(base, 0.6, colored, 0.4, 0)
        else:
            overlay = cv2.addWeighted(base, 0.6, mask, 0.4, 0)
        return overlay

    def _save(self):
        if self.result is None:
            messagebox.showwarning("Attention", "Aucun masque à sauvegarder.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.result)
            messagebox.showinfo("Sauvegardé", f"Masque enregistré :\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 4 – Évaluation & Benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TabEvaluation(tk.Frame):
    def __init__(self, master, shared):
        super().__init__(master, bg=BG)
        self.shared = shared

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, width=280)
        ctrl.pack(side="left", fill="y", padx=(16, 8), pady=16)
        ctrl.pack_propagate(False)

        section_label(ctrl, "Évaluation ponctuelle")
        tk.Label(ctrl, text="Charger un masque vérité terrain\npour calculer les métriques.",
                 bg=BG, fg=FG2, font=FONT_SM, justify="left").pack(anchor="w", padx=16)
        styled_btn(ctrl, "Charger masque GT", self._load_gt, BTN_BG).pack(anchor="w", padx=16, pady=6)
        styled_btn(ctrl, "Calculer métriques", self._eval_single, ACCENT).pack(anchor="w", padx=16, pady=2)

        separator(ctrl)
        info = tk.Frame(ctrl, bg=BG2)
        info.pack(fill="x", padx=16, pady=4)
        self.var_iou  = label_val(info, "IoU   :", 0)
        self.var_dice = label_val(info, "Dice  :", 1)
        self.var_prec = label_val(info, "Prec. :", 2)
        self.var_rec  = label_val(info, "Recall:", 3)
        self.var_f1   = label_val(info, "F1    :", 4)

        separator(ctrl)
        section_label(ctrl, "Benchmark multi-bruit")
        tk.Label(ctrl, text="Type de bruit :", bg=BG, fg=FG2, font=FONT_SM).pack(anchor="w", padx=16)
        self.bench_noise = tk.StringVar(value="gaussian")
        for txt, val in [("Gaussien", "gaussian"), ("Sel-et-Poivre", "salt_and_pepper")]:
            tk.Radiobutton(ctrl, text=txt, variable=self.bench_noise, value=val,
                           bg=BG, fg=FG, selectcolor=ACCENT, activebackground=BG,
                           font=FONT).pack(anchor="w", padx=16)

        styled_btn(ctrl, "Lancer le benchmark", self._run_benchmark, ACCENT).pack(anchor="w", padx=16, pady=8)
        styled_btn(ctrl, "Exporter CSV", self._export_csv, BTN_BG).pack(anchor="w", padx=16, pady=2)

        self.status_var = tk.StringVar(value="En attente…")
        tk.Label(ctrl, textvariable=self.status_var, bg=BG, fg=FG2,
                 font=FONT_SM, wraplength=240, justify="left").pack(anchor="w", padx=16, pady=6)

        self.progress = ttk.Progressbar(ctrl, mode="indeterminate", length=220)
        self.progress.pack(anchor="w", padx=16, pady=4)

        # ── Graphe ────────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=16)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.patch.set_facecolor("#1e1e2e")
        for ax in self.axes.flat:
            ax.set_facecolor("#2a2a3e")
            for sp in ax.spines.values():
                sp.set_color("#3b3b5c")
            ax.tick_params(colors=FG2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._gt    = None
        self._df    = None

    def _load_gt(self):
        path = filedialog.askopenfilename(
            title="Charger masque vérité terrain",
            filetypes=[("Images", "*.png *.jpg *.bmp"), ("Tous", "*.*")]
        )
        if not path:
            return
        gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            messagebox.showerror("Erreur", "Impossible de lire le masque.")
            return
        self._gt = gt
        messagebox.showinfo("OK", "Masque GT chargé.")

    def _eval_single(self):
        pred = self.shared.get("mask")
        if pred is None:
            messagebox.showwarning("Attention", "Faites d'abord une segmentation (onglet 3).")
            return
        if self._gt is None:
            messagebox.showwarning("Attention", "Chargez d'abord un masque vérité terrain.")
            return
        gt_r = cv2.resize(self._gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY) if pred.ndim == 3 else pred
        m = evaluate_mask(pred_gray, gt_r)
        self.var_iou.set(f"{m['iou']:.4f}")
        self.var_dice.set(f"{m['dice']:.4f}")
        self.var_prec.set(f"{m['precision']:.4f}")
        self.var_rec.set(f"{m['recall']:.4f}")
        self.var_f1.set(f"{m['f1']:.4f}")

    def _run_benchmark(self):
        orig = self.shared.get("original")
        if orig is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image (onglet 1).")
            return
        if self._gt is None:
            messagebox.showinfo("Info",
                "Pas de masque GT disponible. Benchmark effectué avec un masque synthétique.")
            gt = self._make_synthetic_gt(orig)
        else:
            gt = cv2.resize(self._gt, (orig.shape[1], orig.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        self.status_var.set("Benchmark en cours…")
        self.progress.start(12)

        def _worker():
            pipelines = {
                "Canny brut":    {"segmentor": lambda img: segment_classical(img, "canny"),
                                   "denoising_method": None},
                "Canny+Médian":  {"segmentor": lambda img: segment_classical(img, "canny"),
                                   "denoising_method": "median"},
                "Otsu brut":     {"segmentor": lambda img: segment_classical(img, "otsu"),
                                   "denoising_method": None},
                "Otsu+Médian":   {"segmentor": lambda img: segment_classical(img, "otsu"),
                                   "denoising_method": "median"},
                "Otsu+NLM":      {"segmentor": lambda img: segment_classical(img, "otsu"),
                                   "denoising_method": "nlm"},
            }
            ntype = self.bench_noise.get()
            levels = [0, 10, 25, 50, 75] if ntype == "gaussian" else [0, 0.02, 0.05, 0.10, 0.20]

            df = benchmark_noise_levels([orig], [gt], pipelines, noise_type=ntype, noise_levels=levels)
            self._df = df
            self.after(0, lambda: self._on_benchmark_done(df, ntype))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_benchmark_done(self, df, ntype):
        self.progress.stop()
        self.status_var.set("Benchmark terminé ✓")

        for ax in self.axes.flat:
            ax.clear()
            ax.set_facecolor("#2a2a3e")
            for sp in ax.spines.values():
                sp.set_color("#3b3b5c")
            ax.tick_params(colors=FG2, labelsize=7)

        metrics = ["iou", "dice", "precision", "recall"]
        titles  = ["IoU", "Dice", "Précision", "Rappel"]
        colors  = [ACCENT, "#22c55e", WARNING, DANGER]
        xlabel  = "σ (Gaussien)" if ntype == "gaussian" else "densité d (S&P)"

        for ax, metric, title, col in zip(self.axes.flat, metrics, titles, colors):
            agg = df.groupby(["pipeline", "noise_level"])[metric].mean().reset_index()
            for pipe, grp in agg.groupby("pipeline"):
                ax.plot(grp["noise_level"], grp[metric], marker="o", markersize=4,
                        label=pipe, linewidth=1.5)
            ax.set_title(title, color=FG2, fontsize=9)
            ax.set_xlabel(xlabel, color=FG2, fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6, facecolor="#2a2a3e", labelcolor=FG2, framealpha=0.8)
            ax.grid(True, alpha=0.2, color="#3b3b5c")

        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def _make_synthetic_gt(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, gt = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gt

    def _export_csv(self):
        if self._df is None:
            messagebox.showwarning("Attention", "Lancez d'abord le benchmark.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if path:
            self._df.to_csv(path, index=False)
            messagebox.showinfo("Exporté", f"CSV enregistré :\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fenêtre principale
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segmentation d'images en présence de bruit  –  PFE 2025/2026")
        self.configure(bg=BG)
        self.geometry("1200x760")
        self.minsize(900, 600)

        # état partagé entre onglets
        self.shared = {
            "original": None,
            "noisy":    None,
            "denoised": None,
            "mask":     None,
        }

        self._build_header()
        self._build_tabs()
        self._apply_ttk_style()

    def _build_header(self):
        hdr = tk.Frame(self, bg=ACCENT, height=48)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text="  Segmentation d'images en présence de bruit ",
                 bg=ACCENT, fg="white", font=("Segoe UI", 11, "bold")).pack(side="left", padx=12, pady=10)

    def _build_tabs(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.TNotebook",       background=BG,  borderwidth=0)
        style.configure("Custom.TNotebook.Tab",
                        background=BG2, foreground=FG2,
                        padding=[18, 8], font=("Segoe UI", 10, "bold"),
                        borderwidth=0)
        style.map("Custom.TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(self, style="Custom.TNotebook")
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        tabs = [
            ("  Bruit",         TabNoise),
            ("  Débruitage",    TabDenoise),
            ("  Segmentation",  TabSegmentation),
            ("  Évaluation",    TabEvaluation),
        ]
        for title, cls in tabs:
            frame = cls(nb, self.shared)
            nb.add(frame, text=title)

    def _apply_ttk_style(self):
        style = ttk.Style()
        style.configure("TProgressbar", troughcolor=BG2, background=ACCENT, thickness=6)
        style.configure("TSeparator",   background="#3b3b5c")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
