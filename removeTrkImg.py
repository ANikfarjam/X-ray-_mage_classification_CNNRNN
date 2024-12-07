from PIL import Image
import tqdm
import pandas as pd
import os
import cv2
import numpy as np


#repair image
def repairImage(img_path):
    img = cv2.imread(img_path)
    try:
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[np.where(img == 0)] = 255  # Create a mask for the truncated area
        inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        save_path = os.path.splitext(img_path)[0] + '_repaired.jpg' 
        success = cv2.imwrite(save_path, inpainted_img)
        os.remove(img_path)
        if success:
            print(f"Repaired image saved to: {save_path}")
            return True
        else:
            print(f"Failed to save the repaired image to: {save_path}")
            return False
    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
        return False
    except cv2.error as cv_error:
        print(f"OpenCV error: {cv_error}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False
        
#check for truncation
def check_images(directory, desc):
    data_dic = {
        'file_path': [],
        'truncated': [],
        'repaired': []
    }

    files_to_check = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith('.jpg')
    ]

    for file_path in tqdm.tqdm(files_to_check, total=len(files_to_check), desc=desc):
        if file_path.endswith('.txt'):
            continue  # Skip .txt files explicitly

        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify if the image is valid
                data_dic['file_path'].append(file_path)
                data_dic['truncated'].append(False)
                data_dic['repaired'].append(False)
        except (IOError, SyntaxError):  # Handle invalid image formats
            data_dic['file_path'].append(file_path)
            data_dic['truncated'].append(True)
            # Attempt to repair the image
            repaired = repairImage(file_path)
            data_dic['repaired'].append(repaired)

    return pd.DataFrame(data_dic)
            
def main():
    """
    DataFrame Example
    name_df=check_images('directory', "Classifying truncated images")
    name_csv = name_df.to_csv('Truncated Images', index=False)
    print(name_csv)

    What do we about files that not images?
    """

    # Legs & Feet
    legAndFeet_dir = 'newData/legsANDfeet'
    legsAndFeet_df = check_images(legAndFeet_dir, 'Repairing legs and feet truncated images')
    
    legsAndFeet_df.to_csv('Legs and Feet Images', index = False)
    print(f"""CSV Data created for Legs & Feet
          {print(legsAndFeet_df.head())}
          Number of Truncated Data: {print(legsAndFeet_df.truncated.value_counts())}
          Number of Repaired Data: {print(legsAndFeet_df.repaired.value_counts())}
          """)

    # Hands and Fingers
    handAndFinger_dir = 'newData/handsNfingers'
    handAndFinger_df = check_images(handAndFinger_dir, 'Repairing hands and fingers truncated images')

    handAndFinger_df.to_csv('Hands and Fingers Images', index = False)
    print(f"""CSV Data created for Hands & Fingers
          {print(handAndFinger_df.head())},
          
          Number of Truncated Images: {print(handAndFinger_df.truncated.value_counts())}
          Number of Repaired Images: {print(handAndFinger_df.repaired.value_counts())}
          """)
    
    # Legs
    """
    leg_dir = 'newData/legs'
    leg_df = check_images(leg_dir, 'Repairing legs truncated images')

    leg_df.to_csv('Legs and Feet Images', index = False)
    print('CSV Data created for Legs')
    """

    # Hands & Forearms
    handAndForearm_dir = 'newData/handNDforearm(posibly Not)'
    handAndForearm_df = check_images(handAndForearm_dir, 'Repairing hands and forearms truncated images')
    handAndForearm_df.to_csv('Hands and forearms Images', index = False)
    print(f"""CSV Data created for Neck and Spine
          {print(handAndForearm_df.head())},
          
          Number of Truncated Images: {print(handAndForearm_df.truncated.value_counts())}
          Number of Repaired Images: {print(handAndForearm_df.repaired.value_counts())}
          """)

    # Neck & Spine
    neckAndSpine_dir = 'newData/NeckSpine'
    neckAndSpine_df = check_images(neckAndSpine_dir, 'Reparing neck and spine truncated images')
    neckAndSpine_df.to_csv('Neck & Spine Images', index = False)
    print(f"""CSV Data created for Neck and Spine
          {print(neckAndSpine_df.head())},
          
          Number of Truncated Images: {print(neckAndSpine_df.truncated.value_counts())}
          Number of Repaired Images: {print(neckAndSpine_df.repaired.value_counts())}
          """)

if __name__== '__main__':
    main()

    