import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermarking with DCT")
        self.root.geometry("800x600")

        self.image_path_embed = ""
        self.watermark_path = ""
        self.image_path_extract = ""

        # Frame for embedding watermark
        self.embed_frame = LabelFrame(root, text="Embed Watermark", padx=10, pady=10)
        self.embed_frame.grid(row=0, column=0, padx=10, pady=10)

        self.label1 = Label(self.embed_frame, text="Original Image")
        self.label1.grid(row=0, column=0)
        self.img_label = Label(self.embed_frame)
        self.img_label.grid(row=1, column=0)

        self.label2 = Label(self.embed_frame, text="Watermark Image")
        self.label2.grid(row=0, column=1)
        self.watermark_label = Label(self.embed_frame)
        self.watermark_label.grid(row=1, column=1)

        self.load_img_btn = Button(self.embed_frame, text="Load Image", command=self.load_image_embed)
        self.load_img_btn.grid(row=2, column=0)

        self.load_watermark_btn = Button(self.embed_frame, text="Load Watermark", command=self.load_watermark)
        self.load_watermark_btn.grid(row=2, column=1)

        self.embed_btn = Button(self.embed_frame, text="Embed Watermark", command=self.embed_watermark)
        self.embed_btn.grid(row=3, column=0, columnspan=2)

        # Frame for extracting watermark
        self.extract_frame = LabelFrame(root, text="Extract Watermark", padx=10, pady=10)
        self.extract_frame.grid(row=0, column=1, padx=10, pady=10)

        self.label3 = Label(self.extract_frame, text="Watermarked Image")
        self.label3.grid(row=0, column=0)
        self.watermarked_img_label = Label(self.extract_frame)
        self.watermarked_img_label.grid(row=1, column=0)

        self.load_watermarked_img_btn = Button(self.extract_frame, text="Load Watermarked Image", command=self.load_image_extract)
        self.load_watermarked_img_btn.grid(row=2, column=0)

        self.extract_btn = Button(self.extract_frame, text="Extract Watermark", command=self.extract_watermark)
        self.extract_btn.grid(row=3, column=0)

        self.extracted_watermark_label = Label(self.extract_frame)
        self.extracted_watermark_label.grid(row=4, column=0)

    def load_image_embed(self):
        self.image_path_embed = filedialog.askopenfilename()
        if self.image_path_embed:
            img = Image.open(self.image_path_embed)
            img.thumbnail((300, 300))
            self.img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.img)

    def load_watermark(self):
        self.watermark_path = filedialog.askopenfilename()
        if self.watermark_path:
            watermark = Image.open(self.watermark_path)
            watermark.thumbnail((100, 100))
            self.watermark_img = ImageTk.PhotoImage(watermark)
            self.watermark_label.config(image=self.watermark_img)

    def embed_watermark(self):
        if not self.image_path_embed or not self.watermark_path:
            messagebox.showerror("Error", "Please load both images")
            return

        img = cv2.imread(self.image_path_embed)
        watermark = cv2.imread(self.watermark_path, cv2.IMREAD_UNCHANGED)

        if watermark.shape[2] == 4:
            watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2BGR)

        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(ycrcb_img)

        watermark_resized = cv2.resize(watermark, (y_channel.shape[1], y_channel.shape[0]))

        img_dct = cv2.dct(np.float32(y_channel))
        watermark_dct = cv2.dct(np.float32(cv2.cvtColor(watermark_resized, cv2.COLOR_BGR2GRAY)))

        alpha = 0.01
        watermarked_dct = img_dct + alpha * watermark_dct
        watermarked_y = cv2.idct(watermarked_dct)
        watermarked_y = np.uint8(watermarked_y)

        watermarked_img = cv2.merge([watermarked_y, cr, cb])
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_YCrCb2BGR)

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, watermarked_img)
            messagebox.showinfo("Success", f"Watermark embedded and saved as {save_path}")

    def load_image_extract(self):
        self.image_path_extract = filedialog.askopenfilename()
        if self.image_path_extract:
            img = Image.open(self.image_path_extract)
            img.thumbnail((300, 300))
            self.watermarked_img = ImageTk.PhotoImage(img)
            self.watermarked_img_label.config(image=self.watermarked_img)

    def extract_watermark(self):
        if not self.image_path_extract or not self.image_path_embed:
            messagebox.showerror("Error", "Please load the watermarked image and the original image")
            return

        watermarked_img = cv2.imread(self.image_path_extract)
        original_img = cv2.imread(self.image_path_embed)

        watermarked_ycrcb = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)
        original_ycrcb = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)

        y_watermarked, _, _ = cv2.split(watermarked_ycrcb)
        y_original, _, _ = cv2.split(original_ycrcb)

        watermarked_dct = cv2.dct(np.float32(y_watermarked))
        original_dct = cv2.dct(np.float32(y_original))

        alpha = 0.01
        extracted_dct = (watermarked_dct - original_dct) / alpha
        extracted_watermark = cv2.idct(extracted_dct)
        extracted_watermark = np.uint8(extracted_watermark)

        # Convert extracted watermark to RGB
        extracted_watermark_rgb = cv2.merge([extracted_watermark] * 3)

        cv2.imwrite("extracted_watermark.png", extracted_watermark_rgb)

        # Display the extracted watermark in the GUI
        extracted_watermark_img = Image.fromarray(extracted_watermark_rgb)
        extracted_watermark_img.thumbnail((100, 100))
        extracted_watermark_tk = ImageTk.PhotoImage(extracted_watermark_img)
        self.extracted_watermark_label.config(image=extracted_watermark_tk)
        self.extracted_watermark_label.image = extracted_watermark_tk

        messagebox.showinfo("Success", "Watermark extracted and displayed.")

if __name__ == "__main__":
    root = Tk()
    app = WatermarkApp(root)
    root.mainloop()
