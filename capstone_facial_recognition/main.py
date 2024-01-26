import os

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

if __name__ == "__main__":
    # Get user information
    name = get_user_info()
    
    # Get the next available face_id
    face_id = len(get_user_info_list()) + 1

    # Save user information to a text file
    save_to_text_file(name, face_id)

    # Run the face_datasets.py script with the obtained face_id
    run_face_datasets_script(face_id)
