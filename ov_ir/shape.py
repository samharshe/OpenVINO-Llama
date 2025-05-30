# convert_gpt2.py
import openvino as ov
import numpy as np

model = ov.convert_model(
    "gpt2-8bit.tflite",
    input=[1, 64]
)

ov.save_model(model, "gpt2_converted.xml")
print("Model converted successfully")

import xml.etree.ElementTree as ET

def fix_tensor_types(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    changes_made = 0
    
    for elem in root.iter():
        if 'precision' in elem.attrib and elem.attrib['precision'] == 'i64':
            elem.attrib['precision'] = 'i32'
            changes_made += 1
            print(f"Changed precision i64->i32 in {elem.tag}")
        
        if 'element_type' in elem.attrib and elem.attrib['element_type'] == 'i64':
            elem.attrib['element_type'] = 'i32'
            changes_made += 1
            print(f"Changed element_type i64->i32 in {elem.tag}")
    
    tree.write('gpt2_converted_fixed.xml')
    print(f"Total changes made: {changes_made}")
    return changes_made

changes = fix_tensor_types('gpt2_converted.xml')
if changes > 0:
    print("Model fixed and saved as: gpt2_converted_fixed.xml")
else:
    print("No i64 types found - model is already compatible")