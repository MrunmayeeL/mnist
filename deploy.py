import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.optim as optim

# Define the neural network architecture (same as used for training)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(784, 500, 10)
model.load_state_dict(torch.load('D:/FSRCNN/mnist_model.pth', map_location=device))
model.to(device)
model.eval()

# Preprocess the canvas image for MNIST model
def preprocess_image(image):
    image = image.convert('L')  # Convert RGBA to grayscale (1 channel)
    image = image.resize((28, 28))  # Resize to 28x28 (MNIST format)
    image = np.array(image)  # Convert to numpy array

    # Invert and normalize pixel values (MNIST: white digits on black background)
    image = 255 - image  
    image = image / 255.0  

    # Convert to a PyTorch tensor
    image = torch.tensor(image).float()
    image = image.view(1, -1)  # Flatten to (1, 784)
    return image.to(device)

# Predict the digit from the image
def predict_digit(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Perform the mathematical calculation
def perform_calculation(first_digit, operation, second_digit):
    if operation == "+":
        return first_digit + second_digit
    elif operation == "-":
        return first_digit - second_digit
    elif operation == "*":
        return first_digit * second_digit
    elif operation == "/":
        return first_digit / second_digit if second_digit != 0 else "Error"

# Streamlit UI
st.title("MNIST Calculator 🧮")
st.write("Draw a digit on the canvas and choose an operation:")

# Canvas for digit input
canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Process canvas image if available
if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype(np.uint8))

    if st.button("Predict"):
        predicted_digit = predict_digit(image)
        st.write(f"**Predicted Digit:** {predicted_digit}")

        # Select operation
        operation = st.radio('Select an operation:', ('+', '-', '*', '/'))

        if st.button("Calculate"):
            result = perform_calculation(predicted_digit, operation, predicted_digit)
            st.write(f"**Calculation Result:** {predicted_digit} {operation} {predicted_digit} = {result}")

# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import torch.optim as optim

# # Define the neural network architecture (same as used for training)
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         return out

# model = NeuralNet(784, 500, 10)  
# model.load_state_dict(torch.load('D:/FSRCNN/mnist_model.pth'))  
# model.eval()  

# def preprocess_image(image):
#     image = image.convert('L')  
#     image = image.resize((28, 28))  
#     image = np.array(image)  
#     image = 255 - image  
#     image = image / 255.0 
#     image = torch.tensor(image).float()  
#     image = image.view(-1, 28 * 28) 
#     return image

# def predict_digit(image):
#     image = preprocess_image(image)  
#     image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
#     with torch.no_grad():  
#         output = model(image)
#         _, predicted = torch.max(output, 1) 
#         return predicted.item()  


# def perform_calculation(first_digit, operation, second_digit):
#     if operation == "+":
#         return first_digit + second_digit
#     elif operation == "-":
#         return first_digit - second_digit
#     elif operation == "*":
#         return first_digit * second_digit
#     elif operation == "/":
#         return first_digit / second_digit if second_digit != 0 else "Error"

# st.title("MNIST Calculator")
# st.write("Draw digits and operations (+, -, *, /) in the canvas below:")


# canvas_result = st_canvas(
#     width=280, height=280,
#     drawing_mode="freedraw",
#     key="canvas"
# )


# if canvas_result.image_data is not None:
    
#     image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    
#     if st.button("Predict"):
#         predicted_digit = predict_digit(image)
#         st.write(f"Predicted Digit: {predicted_digit}")
        
#         st.write("Select an operation:")
#         operation = st.radio('Operation:', ('+', '-', '*', '/'))
        
        
#         if st.button("Calculate"):
    
#             first_digit = predicted_digit
#             second_digit = predicted_digit  
#             result = perform_calculation(first_digit, operation, second_digit)
#             st.write(f"Result: {result}")
