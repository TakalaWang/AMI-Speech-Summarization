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


def get_role(role_file):
    tree = ET.parse(role_file)
    root = tree.getroot()
    roles = {}
    for meeting in root.findall("meeting"):
        role = {}
        for speaker in meeting.findall("speaker"):
            agent = speaker.get("nxt_agent")
            role[agent] = speaker.get("role")
        roles[meeting.get("observation")] = role
    return roles

def get_dialogs(word_file, dialog_file):
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
    dialogs = []
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
            dialogs.append({
                "id": id,
                "agent": agent,
                "starttime": float(starttime),
                "dialog": dialog
            })
   
    return dialogs
    


if __name__ == "__main__":
    split_path = "/datas/store162/takala/ami/dataset/split"
    annotation_dir_path = "/datas/store162/takala/ami/annotation"
    save_dir_path = "/datas/store162/takala/ami/dataset/dialog"
    
    split = get_split(split_path)
    roles = get_role(os.path.join(annotation_dir_path, "corpusResources", "meetings.xml"))
    for split_name, meetings in split.items():
        os.makedirs(os.path.join(save_dir_path, split_name), exist_ok=True)
        
        for meeting in meetings:
            all_dialogs = []
            for agent in ["A", "B", "C", "D"]:
                dialogs = get_dialogs(
                    os.path.join(annotation_dir_path, "words", f"{meeting}.{agent}.words.xml"), 
                    os.path.join(annotation_dir_path, "dialogueActs", f"{meeting}.{agent}.dialog-act.xml")
                )
                all_dialogs.extend(dialogs)
            
            all_dialogs.sort(key=lambda x: x["starttime"])
            
            with open(os.path.join(save_dir_path, split_name, meeting), "w") as f:
                for dialog in all_dialogs:
                    agent = dialog['agent']
                    id = dialog['id']
                    name = roles[meeting][agent]
                    starttime = dialog['starttime']
                    dialog = dialog['dialog']
                    
                    f.write(f"{name}: {dialog}\n")
