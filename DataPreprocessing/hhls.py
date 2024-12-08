import os
#hand
fracHandDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Hand/fractured'
FHandFiles = [os.path.join(root, f) for root, _, files in os.walk(fracHandDir) for f in files]

nonfracHandDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Hand/nonfractured'
NFHandFiles = [os.path.join(root, f) for root, _, files in os.walk(nonfracHandDir) for f in files]



# Hip 
fracHipDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Hip/fractured'
FHipFiles = [os.path.join(root, f) for root, _, files in os.walk(fracHipDir) for f in files]

nonFracHipDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Hip/nonfractured'
NFHipFiles = [os.path.join(root, f) for root, _, files in os.walk(nonFracHipDir) for f in files]

# Leg

fracLegDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Leg/fractured'
FLegFiles = [os.path.join(root, f) for root, _, files in os.walk(fracLegDir) for f in files]
#print(f"Total Fractured Hip X-Rays: {len(FLegFiles)}")

nonFracLegDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Leg/nonfractured'
NFLegFiles = [os.path.join(root, f) for root, _, files in os.walk(nonFracLegDir) for f in files]

# Shoulder

fracShoulderDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Shoulder/fractured'
FShoulderFiles = [os.path.join(root, f) for root, _, files in os.walk(fracShoulderDir) for f in files]
print(f"Total Fractured Hip X-Rays: {len(FLegFiles)}")

nonFracShoulderDir = '../Data/Data/FracAtlas_Hand_Leg_Hip_Shoulder/images/Shoulder/nonfractured'
NFShoulderFiles = [os.path.join(root, f) for root, _, files in os.walk(nonFracShoulderDir) for f in files]

# Mixed 


#prin the result
print(f"""
        Total Fractured Hand X-Rays: {len(FHandFiles)}
        Total Non Fractured Hand X-Rays: {len(NFHandFiles)}
        
        Total Fractured Hip X-Rays: {len(FHipFiles)}
        Total Non Fractured Hip X-Rays: {len(NFHipFiles)}
        
        Total Fractured Shoulder X-Rays: {len(FShoulderFiles)}
        Total Non Fractured Shoulder X-Rays: {len(NFShoulderFiles)}
        
        Total Fractured Leg X-Rays: {len(FLegFiles)}
        Total Non Fractured Leg X-Rays: {len(NFLegFiles)}
        """)


# To Do List:
# Get me the total of train/test/val of hand part 2

# Train
handP2Train_frac = '../Data/Data/Hand_Part_2/train/fractured'
handP2Train_frac_files = [os.path.join(root, f) for root, _, files in os.walk(handP2Train_frac) for f in files]

handP2Train_nfrac = '../Data/Data/Hand_Part_2/train/nonfractured'
handP2Train_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(handP2Train_nfrac) for f in files]

# Test
handP2Test_frac = '../Data/Data/Hand_Part_2/test/fractured'
handP2Test_frac_files = [os.path.join(root, f) for root, _, files in os.walk(handP2Test_frac) for f in files]

handP2Test_nfrac = '../Data/Data/Hand_Part_2/test/nonfractured'
handP2Test_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(handP2Test_nfrac) for f in files]

# Valid
handP2Valid_frac = '../Data/Data/Hand_Part_2/valid/fractured'
handP2Valid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(handP2Valid_frac) for f in files]

handP2Valid_nfrac = '../Data/Data/Hand_Part_2/valid/nonfractured'
handP2Valid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(handP2Valid_nfrac) for f in files]


# Get me the total of train/test/val of leg

# Train
LegTrain_frac = '../Data/Data/Legs/train/fractured'
LegTrain_frac_files = [os.path.join(root, f) for root, _, files in os.walk(LegTrain_frac) for f in files]

LegTrain_nfrac = '../Data/Data/Legs/train/nonfractured'
LegTrain_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(LegTrain_nfrac) for f in files]

# Test
LegTest_frac = '../Data/Data/Legs/test/fractured'
LegTest_frac_files = [os.path.join(root, f) for root, _, files in os.walk(LegTest_frac) for f in files]

LegTest_nfrac = '../Data/Data/Legs/test/nonfractured'
LegTest_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(LegTest_nfrac) for f in files]

# Valid
LegValid_frac = '../Data/Data/Legs/valid/fractured'
LegValid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(LegValid_frac) for f in files]

LegValid_nfrac = '../Data/Data/Legs/valid/nonfractured'
LegValid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(LegValid_nfrac) for f in files]

# do the same for shoulder if there is any

# Train
ShoulderTrain_frac = '../Data/Data/Shoulders/train/fractured'
ShoulderTrain_frac_files = [os.path.join(root, f) for root, _, files in os.walk(ShoulderTrain_frac) for f in files]

ShoulderTrain_nfrac = '../Data/Data/Shoulders/train/nonfractured'
ShoulderTrain_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(ShoulderTrain_nfrac) for f in files]

# Test
ShoulderTest_frac = '../Data/Data/Shoulders/test/fractured'
ShoulderTest_frac_files = [os.path.join(root, f) for root, _, files in os.walk(ShoulderTest_frac) for f in files]

ShoulderTest_nfrac = '../Data/Data/Shoulders/test/nonfractured'
ShoulderTest_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(ShoulderTest_nfrac) for f in files]

# Valid
ShoulderValid_frac = '../Data/Data/Shoulders/valid/fractured'
ShoulderValid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(ShoulderValid_frac) for f in files]

ShoulderValid_nfrac = '../Data/Data/Shoulders/valid/nonfractured'
ShoulderValid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(ShoulderValid_nfrac) for f in files]

# Legs&Feet == LF

# Train
LFTrain_frac = '../Data/Data/Legs&Feet/train/fractured'
LFTrain_frac_files = [os.path.join(root, f) for root, _, files in os.walk(LFTrain_frac) for f in files]

LFTrain_nfrac = '../Data/Data/Legs&Feet/train/nonfractured'
LFTrain_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(LFTrain_nfrac) for f in files]

# Test
LFTest_frac = '../Data/Data/Legs&Feet/test/fractured'
LFTest_frac_files = [os.path.join(root, f) for root, _, files in os.walk(LFTest_frac) for f in files]

LFTest_nfrac = '../Data/Data/Legs&Feet/test/nonfractured'
LFTest_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(LFTest_nfrac) for f in files]

# Valid
LFValid_frac = '../Data/Data/Legs&Feet/valid/fractured'
LFValid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(LFValid_frac) for f in files]

LFValid_nfrac = '../Data/Data/Legs&Feet/valid/nonfractured'
LFValid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(LFValid_nfrac) for f in files]

# Hands & Forearms

# Train
HFTrain_frac = '../Data/Data/Hands&Forearms/training/fractured'
HFTrain_frac_files = [os.path.join(root, f) for root, _, files in os.walk(HFTrain_frac) for f in files]

HFTrain_nfrac = '../Data/Data/Hands&Forearms/training/not_fractured'
HFTrain_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(HFTrain_nfrac) for f in files]

# Test
HFTest_frac = '../Data/Data/Hands&Forearms/testing/fractured'
HFTest_frac_files = [os.path.join(root, f) for root, _, files in os.walk(HFTest_frac) for f in files]

HFTest_nfrac = '../Data/Data/Hands&Forearms/testing/not_fractured'
HFTest_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(HFTest_nfrac) for f in files]

# Valid
HFValid_frac = '../Data/Data/Hands&Forearms/validation/fractured'
HFValid_frac_files = [os.path.join(root, f) for root, _, files in os.walk(HFValid_frac) for f in files]

HFValid_nfrac = '../Data/Data/Hands&Forearms/validation/not_fractured'
HFValid_nfrac_files = [os.path.join(root, f) for root, _, files in os.walk(HFValid_nfrac) for f in files]

# Other Hand This hand is not labeled fractured or non fractured but simply other and hand

# Train
HandTrain_hand = '../Data/Data/Hand/train/hand'
HandTrain_hand_files = [os.path.join(root, f) for root, _, files in os.walk(HandTrain_hand) for f in files]

HandTrain_other = '../Data/Data/Hand/train/other'
HandTrain_other_files = [os.path.join(root, f) for root, _, files in os.walk(HandTrain_other) for f in files]

# Test
HandTest_hand = '../Data/Data/Hand/test/hand'
HandTest_hand_files = [os.path.join(root, f) for root, _, files in os.walk(HandTest_hand) for f in files]

HandTest_other = '../Data/Data/Hand/test/other'
HandTest_other_files = [os.path.join(root, f) for root, _, files in os.walk(HandTest_other) for f in files]

# Valid
HandValid_hand = '../Data/Data/Hand/val/hand'
HandValid_hand_files = [os.path.join(root, f) for root, _, files in os.walk(HandValid_hand) for f in files]

HandValid_other = '../Data/Data/Hand/val/other'
HandValid_other_files = [os.path.join(root, f) for root, _, files in os.walk(HandValid_other) for f in files]


# Print out all the info 

print(f"""
        # Hand & Other (Not Labeled Fractured or Non Fractured)
        
        Train Hand Files: {len(HandTrain_hand_files)}
        Train Other Files: {len(HandTrain_other_files)}
        
        Test Hand Files: {len(HandTest_hand_files)}
        Test Other Files: {len(HandTest_other_files)}
        
        Valid Hand Files: {len(HandValid_hand_files)}
        Valid Other Files: {len(HandValid_other_files)}
        
        # Hand Part 2
       
        Train Frac Files: {len(handP2Train_frac_files)}
        Train NonFrac Files: {len(handP2Train_nfrac_files)}
       
        Test Frac Files: {len(handP2Test_frac_files)}
        Test NonFrac Files: {len(handP2Test_nfrac_files)}
       
        Valid Frac Files: {len(handP2Valid_frac_files)}
        Valid NonFrac Files: {len(handP2Valid_nfrac_files)}
        
        # Hands & Forearms
        
        Train Frac Files: {len(HFTrain_frac_files)}
        Train NonFrac Files: {len(HFTrain_nfrac_files)}
       
        Test Frac Files: {len(HFTest_frac_files)}
        Test NonFrac Files: {len(HFTest_nfrac_files)}
       
        Valid Frac Files: {len(HFValid_frac_files)}
        Valid NonFrac Files: {len(HFValid_nfrac_files)}
        
        # Legs
        
        Train Frac Files: {len(LegTrain_frac_files)}
        Train NonFrac Files: {len(LegTrain_nfrac_files)}
       
        Test Frac Files: {len(LegTest_frac_files)}
        Test NonFrac Files: {len(LegTest_nfrac_files)}
       
        Valid Frac Files: {len(LegValid_frac_files)}
        Valid NonFrac Files: {len(LegValid_nfrac_files)}
        
        # Legs&Feet
        
        Train Frac Files: {len(LFTrain_frac_files)}
        Train NonFrac Files: {len(LFTrain_nfrac_files)}
       
        Test Frac Files: {len(LFTest_frac_files)}
        Test NonFrac Files: {len(LFTest_nfrac_files)}
       
        Valid Frac Files: {len(LFValid_frac_files)}
        Valid NonFrac Files: {len(LFValid_nfrac_files)}
        
        # Shoulders
        
        Train Frac Files: {len(ShoulderTrain_frac_files)}
        Train NonFrac Files: {len(ShoulderTrain_nfrac_files)}
        
        Test Frac Files: {len(ShoulderTest_frac_files)}
        Test NonFrac Files: {len(ShoulderTest_nfrac_files)}
       
        Valid Frac Files: {len(ShoulderValid_frac_files)}
        Valid NonFrac Files: {len(ShoulderValid_nfrac_files)}
     
        """)
