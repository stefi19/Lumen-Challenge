import easyocr
import cv2


def extract_numbers_from_image(image_path):
    # Load image and initialize reader
    reader = easyocr.Reader(['en'])

    # Run OCR
    results = reader.readtext(image_path)

    # Extract texts with digits
    numbers = []
    for (bbox, text, prob) in results:
        if any(char.isdigit() for char in text):
            numbers.append(text)

    return numbers

def extract_number(numbers):
    for num in numbers:
        if len(num) < 3:
            return num

image_path = 'bus.jpg'  # replace with your actual path
numbers = extract_numbers_from_image(image_path)
print("Numbers detected:", extract_number(numbers))
