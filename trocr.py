import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


device = "cuda" if torch.cuda.is_available() else "cpu"


class IAMDataset(Dataset):
    def __init__(self, crops, processor):
        self.crops = crops
        self.processor = processor

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crp = self.crops[idx]
        pixel_values = self.processor(crp, return_tensors="pt").pixel_values
        encoding = {"pixel_values": pixel_values.squeeze()}
        return encoding

def get_processor_model(checkpoint:str):
    rec_processor = TrOCRProcessor.from_pretrained('/home/n13452/saved_models/trocr_printed_processor/')
    rec_model = VisionEncoderDecoderModel.from_pretrained('/home/n13452/saved_models/trocr_printed_model/')
    rec_model.config.eos_token_id = 2
    rec_model.config.pad_token_id = 2
    rec_model.to(device)
    rec_model.eval()
    return rec_processor, rec_model
