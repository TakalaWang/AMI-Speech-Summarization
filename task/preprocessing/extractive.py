import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

def get_split(split_path):
    split = {}
    for split_name in ['train', 'valid', 'test']:
        with open(os.path.join(split_path, f"{split_name}.txt")) as f:
            split[split_name] = f.read().splitlines()
    return split

def get_dialogs_dict(word_file, dialog_file):
    tree = ET.parse(word_file)
    root = tree.getroot()
    
    words = defaultdict(lambda: {"text": "", "starttime": float('inf')})
    for w in root.findall(".//w"):
        if w.text is None or w.get("starttime") is None:
            continue
        words[w.get("{http://nite.sourceforge.net/}id")] = {
            "text": w.text,
            "starttime": float(w.get("starttime")),
        }

    tree = ET.parse(dialog_file)
    root = tree.getroot()
    dialogs = defaultdict(str)
    for dact in root.findall("dact"):
        id = dact.get("{http://nite.sourceforge.net/}id").split(".")[-1]
        child = dact.find("{http://nite.sourceforge.net/}child")    
        href = child.get("href")
        starttime = float('inf')
        dialog = ""
        agent = ""
        if ".." in href:
            pattern = r"^(.*)\.xml#id\((.*)\.words(\d+)\)\.\.id\(\2\.words(\d+)\)$"
            match = re.match(pattern, href)
            base = match.group(1)
            agent = match.group(2).split('.')[-1]
            from_id = match.group(3)
            to_id = match.group(4)
            
            for i in range(int(from_id), int(to_id) + 1):
                if words[f"{base}{str(i)}"]:
                    starttime = min(starttime, words[f"{base}{str(i)}"]["starttime"])
                    dialog += words[f"{base}{str(i)}"]["text"] + " "
        else:
            pattern = r"^(.*)\.xml#id\((.*)\.words(\d+)"
            match = re.match(pattern, href)
            agent = match.group(2).split('.')[-1]
            dialog = words[match.group(3)]["text"]
            starttime = min(starttime, words[match.group(3)]["starttime"])
            
        dialog = dialog.strip()
        if dialog:
            dialogs[f"{agent}.{id}"] = dialog
   
    return dialogs


# <nite:root nite:id="ES2002a.extsumm" xmlns:nite="http://nite.sourceforge.net/">
#    <extsumm nite:id="ES2002a.extsumm.dharshi.1">
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.3)"/>
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.12)"/>
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.19)"/>
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.22)"/>
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.28)"/>
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.31)"/>
#       <nite:child href="ES2002a.B.dialog-act.xml#id(ES2002a.B.dialog-act.dharshi.33)"/>
#       <nite:child href="ES2002a.D.dialog-act.xml#id(ES2002a.D.dialog-act.dharshi.16)..id(ES2002a.D.dialog-act.dharshi.17)"/>
#       <nite:child href="ES2002a.D.dialog-act.xml#id(ES2002a.D.dialog-act.dharshi.23)..id(ES2002a.D.dialog-act.dharshi.24)"/>
def get_extractive(extractive_file, dialog_file):
    tree = ET.parse(extractive_file)
    root = tree.getroot()
    extractives = []
    extsum = root.find("extsumm")
    for child in extsum.findall("{http://nite.sourceforge.net/}child"):
        href = child.get("href")
        
        if ".." in href:    
            pattern = r"^(.*)\.xml#id\((.*)\.dialog-act\.(.*)\.(\d+)\)\.\.id\(\2\.dialog-act\.(.*)\.(\d+)\)$"
            match = re.match(pattern, href)
            agent = match.group(2).split('.')[-1]
            from_id = match.group(4)
            to_id = match.group(6)
            
            for i in range(int(from_id), int(to_id) + 1):
                extractive = dialog_file[f"{agent}.{str(i)}"]
                if extractive:
                    extractives.append(extractive)
        else:
            pattern = r"^(.*)\.xml#id\((.*)\.dialog-act\.(.*)\.(\d+)"
            match = re.match(pattern, href)
            agent = match.group(2).split('.')[-1]
            dialog_id = match.group(4)
            extractive = dialog_file[f"{agent}.{dialog_id}"]
            if extractive:
                extractives.append(extractive)
                
    return extractives


if __name__ == "__main__":
    split_path = "/datas/store162/takala/ami/dataset/split"
    annotation_dir_path = "/datas/store162/takala/ami/annotation"
    save_dir_path = "/datas/store162/takala/ami/dataset/extractive"
    
    split = get_split(split_path)
    for split_name, meetings in split.items():
        os.makedirs(os.path.join(save_dir_path, split_name), exist_ok=True)
        
        for meeting in meetings:
            all_dialogs = defaultdict(str)
            for agent in ["A", "B", "C", "D"]:
                dialogs = get_dialogs_dict(
                    os.path.join(annotation_dir_path, "words", f"{meeting}.{agent}.words.xml"), 
                    os.path.join(annotation_dir_path, "dialogueActs", f"{meeting}.{agent}.dialog-act.xml")
                )
                all_dialogs.update(dialogs)
            
            extractives = get_extractive(
                os.path.join(annotation_dir_path, "extractive", f"{meeting}.extsumm.xml"),
                all_dialogs
            )
            
            with open(os.path.join(save_dir_path, split_name, meeting), "w") as f:
                for extractive in extractives:
                    f.write(f"{extractive}\n")
                    
