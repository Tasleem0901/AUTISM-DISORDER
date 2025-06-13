from tkinter import *
from tkinter import simpledialog, filedialog, messagebox
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import joblib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Initialize global variables
dataset = None
X_train, X_test, y_train, y_test = None, None, None, None
label_encoders = []
classifier = None

# GUI Window
main = Tk()
main.title("Autism Spectrum Disorder Detection")
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width-100}x{screen_height-100}")

# Functions
def upload_dataset():
    global dataset
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        dataset = pd.read_csv(filepath)
        messagebox.showinfo("Dataset Upload", f"Dataset uploaded successfully! Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

def processDataset():
    global X, Y, X_train, X_test, y_train, y_test, label_encoders
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)
    columns = dataset.columns
    
    # Initialize label encoders for categorical columns
    label_encoders = []
    for i in range(len(columns)):
        if dataset[columns[i]].dtype == 'object':  # Check for non-numeric data
            le = LabelEncoder()
            dataset[columns[i]] = le.fit_transform(dataset[columns[i]].astype(str))
            label_encoders.append(le)  # Add the encoder to the list
        else:
            label_encoders.append(None)  # For numeric columns

    text.insert(END, str(dataset.head()) + "\n\n")
    dataset_values = dataset.values
    X = dataset_values[:, :-1].astype(float)  # Features
    Y = dataset_values[:, -1].astype(int)    # Target labels

    # Apply SMOTE to balance the dataset
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, Y = smote.fit_resample(X, Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    text.insert(END, f"Total records after SMOTE: {len(X)}\n")
    text.insert(END, f"Training records: {len(X_train)}\n")
    text.insert(END, f"Testing records: {len(X_test)}\n")

def train_cnn():
    global classifier
    if X_train is None or y_train is None:
        messagebox.showerror("Error", "Please preprocess the dataset first.")
        return

    # Reshape for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

    # Building the CNN
    classifier = Sequential()
    classifier.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X_train_cnn.shape[1], 1, 1)))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=len(np.unique(y_train)), activation='softmax'))

    # Compile
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train
    classifier.fit(X_train_cnn, y_train, batch_size=32, epochs=10, validation_data=(X_test_cnn, y_test))

    messagebox.showinfo("Training", "CNN Model trained successfully.")

def evaluate_model():
    if classifier is None:
        messagebox.showerror("Error", "Please train the model first.")
        return

    # Reshape for CNN prediction
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
    y_pred = np.argmax(classifier.predict(X_test_cnn), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    result_text = f"Confusion Matrix:\n{cm}\n\nAccuracy: {acc*100:.2f}%\n\nClassification Report:\n{report}"
    messagebox.showinfo("Evaluation", result_text)

def autism_graph(predict):
    no_autism_count = np.sum(predict == 0)
    autism_count = np.sum(predict == 1)
    
    labels = ["No Autism", "Autism"]
    counts = [no_autism_count, autism_count]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['blue', 'red'])
    plt.xlabel('Prediction Class')
    plt.ylabel('Count')
    plt.title('Prediction Results: No Autism vs Autism')
    plt.show()

def detectAutism():
    global classifier, label_encoders, columns
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset", filetypes=[("CSV Files", "*.csv")])
    if not filename:
        return
    testData = pd.read_csv(filename)
    testData.fillna(0, inplace=True)
    columns = testData.columns
    
    for i in range(len(columns)):
        if label_encoders[i] is not None:
            testData[columns[i]] = label_encoders[i].transform(testData[columns[i]].astype(str))

    testData = testData.values
    X1 = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
    predict = classifier.predict(X1)
    predict = np.argmax(predict, axis=1)
    label = ["No Autism Disorder Detected", "Autism Disorder Detected"]
    
    for i in range(len(predict)):
        text.insert(END, f"Test Data = {testData[i]} =====> Predicted Output : {label[predict[i]]}\n\n")
    
    autism_graph(predict)

# Buttons for GUI
Button(main, text="Upload Dataset", command=upload_dataset).pack(pady=10)
Button(main, text="Preprocess Dataset", command=processDataset).pack(pady=10)
Button(main, text="Train CNN Model", command=train_cnn).pack(pady=10)
Button(main, text="Evaluate Model", command=evaluate_model).pack(pady=10)
Button(main, text="Real-Time Test", command=detectAutism).pack(pady=10)
text = Text(main, height=20, width=120, font=('times', 12))
text.place(x=70, y=300)

main.mainloop()