from classifier import predict_text
from visualize import plot_analysis
import time

print("Welcome to NLP Text Classification Project")
while True:
    print("\nOptions:")
    print("1. Predict Sentiment & Spam")
    print("2. Show Graph Analysis")
    print("3. Exit")
    choice = input("Enter choice: ")

    if choice == '1':
        text = input("Enter your message: ")
        result = predict_text(text)
        print(f"Prediction: {result}")
    elif choice == '2':
        plot_analysis()
    elif choice == '3':
        print("Exiting...")
        time.sleep(1)
        break
    else:
        print("Invalid choice. Try again.")