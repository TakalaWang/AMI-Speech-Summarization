import os
import xml.etree.ElementTree as ET

full_name_map = {
    "PM": "project manager",
    "ME": "marketing expert",
    "ID": "industrial designer",
    "UI": "user interface designe",
}

def get_split(split_path):
    split = {}
    for split_name in ['train', 'valid', 'test']:
        with open(os.path.join(split_path, f"{split_name}.txt")) as f:
            split[split_name] = f.read().splitlines()
    return split

def parse_abstractive_file(abs_file):
    """Parse abstractive summary file"""
    tree = ET.parse(abs_file)
    root = tree.getroot()
    
    sentences = []
    abstract = root.find("abstract")

    for sentence in abstract.findall(".//sentence"):
        sentences.append(f"{sentence.text}\n")
    return sentences

if __name__ == "__main__":
    split_path = "/datas/store162/takala/ami/dataset/split"
    annotation_dir_path = "/datas/store162/takala/ami/annotation"
    
    save_dir_path = "/datas/store162/takala/ami/dataset/abstractive"
    
    split = get_split(split_path)
    for split_name, meetings in split.items():
        os.makedirs(os.path.join(save_dir_path, split_name), exist_ok=True)
        for meeting in meetings:
            abs_file = os.path.join(annotation_dir_path, "abstractive", f"{meeting}.abssumm.xml")
            sentences = parse_abstractive_file(abs_file)
            for name, full_name in full_name_map.items():
                sentences = [sentence.replace(name, full_name) for sentence in sentences]
            with open(os.path.join(save_dir_path, split_name, meeting), "w") as f:
                f.write("".join(sentences))