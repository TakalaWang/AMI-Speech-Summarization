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

def get_word_dict(word_file):
    tree = ET.parse(word_file)
    root = tree.getroot()
    
    words = defaultdict(str)
    for w in root.findall(".//w"):
        words[w.get("{http://nite.sourceforge.net/}id")] = w.text
   
    return words
    

def parse_topic_file(topic_file, words):
    tree = ET.parse(topic_file)
    root = tree.getroot()
    dialogs = []
    for topic in root.findall("topic"):
        for child in topic.findall("{http://nite.sourceforge.net/}child"):
            href = child.get("href")
            
            dialog = ""
            agent = ""
            
            if ".." in href:
                pattern = r"^(.*)\.xml#id\((.*)\.words(\d+)\)\.\.id\(\2\.words(\d+)\)$"
                match = re.match(pattern, href)
                base = match.group(1)
                agent = match.group(2).split('.')[-1]
                from_id = int(match.group(3))
                to_id = int(match.group(4))
                
                if to_id - from_id < 5:
                    continue

                dialog = " ".join([words[f"{base}{str(i)}"] for i in range(from_id, to_id + 1) if words[f"{base}{str(i)}"]])
            else:
                pattern = r"^(.*)\.xml#id\((.*)\.words(\d+)\)$"
                match = re.match(pattern, href)
                base = match.group(1)
                agent = match.group(2).split('.')[-1]
                from_id = int(match.group(3))
                dialog = words[f"{base}{str(from_id)}"]
                
            dialog = dialog.strip()
            if dialog:
                dialogs.append({
                    "agent": agent,
                    "dialog": dialog
                })
    return dialogs

if __name__ == "__main__":
    split_path = "/datas/store162/takala/ami/dataset/split"
    annotation_dir_path = "/datas/store162/takala/ami/annotation"
    save_dir_path = "/datas/store162/takala/ami/dataset/topic"
    
    split = get_split(split_path)
    roles = get_role(os.path.join(annotation_dir_path, "corpusResources", "meetings.xml"))
    for split_name, meetings in split.items():
        os.makedirs(os.path.join(save_dir_path, split_name), exist_ok=True)
        for meeting in meetings:
            all_words = defaultdict(str)
            for agent in ["A", "B", "C", "D"]:
                words = get_word_dict(os.path.join(annotation_dir_path, "words", f"{meeting}.{agent}.words.xml"))
                all_words.update(words)
            if not os.path.exists(os.path.join(annotation_dir_path, "topics", f"{meeting}.topic.xml")):
                print(f"No topic file for {meeting}")
                continue
            topics = parse_topic_file(os.path.join(annotation_dir_path, "topics", f"{meeting}.topic.xml"), all_words)
            
            
            with open(os.path.join(save_dir_path, split_name, meeting), "w") as f:
                for topic in topics:
                    name = roles[meeting][topic['agent']]
                    dialog = topic['dialog']
                    f.write(f"<{name}> {dialog}\n")
