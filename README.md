# Helmet-and-license-plate-detection
This is my project on real time  helmet and license plate detection using YOLOv3(for detection) and pytesseract(for OCR).
This uses a pretrained YOLOv3 model to detect vehicles and classify whether the rider is wearing a hemet or not. And then uses pytesseract API for reading the Characters on the plate.
If the OCR functionality does not give proper result, i added a fearure where it crops out the license plate and gived you the cropped out license plate as a png.
