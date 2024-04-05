import os
import subprocess

def get_user_info():
    name = input("Enter your name: ")
    return name

def get_user_info_list(filename="user_info.txt"):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            user_info_list = []
            current_user = {}
            for line in lines:
                if line.startswith('Name'):
                    current_user['Name'] = line.split(':')[-1].strip()
                elif line.startswith('Face ID'):
                    current_user['Face ID'] = int(line.split(':')[-1].strip())
                    user_info_list.append(current_user.copy())
            return user_info_list
    except FileNotFoundError:
        return []

def save_to_text_file(name, face_id, filename="user_info.txt"):
    user_info_list = get_user_info_list()
    user_info_list.append({"Name": name, "Face ID": face_id})

    with open(filename, 'w') as file:
        for user_info in user_info_list:
            file.write(f"Name: {user_info['Name']}\n")
            file.write(f"Face ID: {user_info['Face ID']}\n")

def run_face_datasets_script(face_id):
    os.system(f"python face_datasets.py {face_id}")

def run_script(script_name):
    # Runs a Python script in a subprocess
    subprocess.run(['python', script_name], check=True)

if __name__ == "__main__":
    # Get user information
    name = get_user_info()
    
    # Get the next available face_id
    face_id = len(get_user_info_list()) + 1

    # Save user information to a text file
    save_to_text_file(name, face_id)
    print("Saving user information")

    # Run the face_datasets.py script with the obtained face_id
    run_face_datasets_script(face_id)
    print("Obtained 100 Image Samples")

    # After collecting the dataset, run training.py to train the model
    run_script('training.py')
    print("Facial Recognition Algorithm Trained")
    # Once training is complete, run face_recognition.py to start face recognition
    run_script('face_recognition.py')
