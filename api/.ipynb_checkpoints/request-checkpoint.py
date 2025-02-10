import requests

url = "http://127.0.0.1:8000/predict/"  # Update this if your API is hosted elsewhere

data = {
    "Cost_of_vehicle": 23600,
    "Min_coverage": 590,
    "Max_coverage": 5978,
    "Expiry_date": "12-04-2025",
    "Insurance_company": "B"
}
image_path = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\images\test_images\img_4538519.jpg"

# Open the file and send it in the request
with open(image_path, "rb") as img_file:
    files = {"files": (image_path, img_file, "image/jpeg")}  # File kept open inside 'with' block
    response = requests.post(url, json=data, files=files)  # Make sure file is open during request

# Print the response
print(response.json())
