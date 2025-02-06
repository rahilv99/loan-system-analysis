import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm
from collections import defaultdict
from pdf2image import convert_from_path
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import time


# Models
from transformers import AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection

class TableParser:
    def __init__(self, lang='en', detection_threshold=0.5, structure_threshold=0.5, crop_padding=5):

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set language
        self.lang = lang
        
        # Load models
        self.detection_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
        self.detection_model.to(self.device)
        
        self.structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
        self.structure_model.to(self.device)
        
        # Load OCR reader
        self.reader = easyocr.Reader([self.lang])
        
        # Thresholds and configuration
        self.detection_class_thresholds = {
            "table": detection_threshold,
            "table rotated": detection_threshold,
            "no object": 10
        }
        self.structure_class_thresholds = {
            "table row": structure_threshold,
            "table column": structure_threshold,
            "no object": 10
        }
        self.crop_padding = crop_padding
        
        # Transforms
        self.detection_transform = transforms.Compose([
            self.MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.structure_transform = transforms.Compose([
            self.MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    class MaxResize:
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
            return resized_image
    
    def box_cxcywh_to_xyxy(self, x):
        """Convert center-based bounding box to corner-based."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        """Rescale bounding boxes to original image size."""
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label, class_thresholds):
        """Convert model outputs to detected objects."""
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if class_label != 'no object' and score >= class_thresholds.get(class_label, 0.5):
                objects.append({
                    'label': class_label, 
                    'score': float(score),
                    'bbox': [float(elem) for elem in bbox]
                })

        return objects

    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=10):
        """Crop detected tables from the image."""
        table_crops = []
        for obj in objects:
            if obj['score'] < class_thresholds[obj['label']]:
                continue

            bbox = obj['bbox']
            bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

            cropped_img = img.crop(bbox)

            table_tokens = [token for token in tokens if self.iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [
                    token['bbox'][0]-bbox[0],
                    token['bbox'][1]-bbox[1],
                    token['bbox'][2]-bbox[0],
                    token['bbox'][3]-bbox[1]
                ]

            # Handle rotated tables
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [
                        cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]
                    ]
                    token['bbox'] = bbox

            table_crops.append({
                'image': cropped_img,
                'tokens': table_tokens
            })

        return table_crops

    def iob(self, bbox1, bbox2):
        """Calculate Intersection over Bounding Box."""
        x11, y11, x12, y12 = bbox1
        x21, y21, x22, y22 = bbox2

        xA = max(x11, x21)
        yA = max(y11, y21)
        xB = min(x12, x22)
        yB = min(y12, y22)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        bbox1Area = (x12 - x11) * (y12 - y11)

        return interArea / bbox1Area if bbox1Area > 0 else 0

    def get_cell_coordinates_by_col(self, table_data):
        """Get cell coordinates by column."""
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']

        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        def find_cell_coordinates(row, column):
            return [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []
        for column in columns:
            col_cells = []
            for row in rows:
                cell_bbox = find_cell_coordinates(row, column)
                col_cells.append({'column': row['bbox'], 'cell': cell_bbox})

            col_cells.sort(key=lambda x: x['column'][1])
            cell_coordinates.append({'col': column['bbox'], 'cells': col_cells, 'cell_count': len(col_cells)})
        cell_coordinates.sort(key=lambda x: x['col'][0])

        return cell_coordinates 

    def apply_ocr(self, cell_coordinates, cropped_table):
        """Apply OCR to cell coordinates."""
        data = dict()
        max_num_columns = 0
        for idx, col in enumerate(cell_coordinates):
            col_text = []
            for cell in col["cells"]:
                # crop cell out of image
                cell_image = np.array(cropped_table.crop(cell["cell"]))
                
                # apply OCR
                result = self.reader.readtext(np.array(cell_image))
                
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                    col_text.append(text)
                else:
                    col_text.append("")
            
            if len(col_text) > max_num_columns:
                max_num_columns = len(col_text)
            
            data[idx] = col_text
        
        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for col, col_data in data.copy().items():
            if len(col_data) != max_num_columns:
                col_data = col_data + ["" for _ in range(max_num_columns - len(col_data))]
            data[col] = col_data
        
        return data

    def parse_pdf_tables(self, pdf_path):
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        tables_dataframes = []

        for image in images:
            # Convert to RGB
            image = image.convert("RGB")

            table_detect_time_start = time.time()

            # Detect tables
            pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.detection_model(pixel_values)

            # Get table objects
            detection_id2label = self.detection_model.config.id2label
            detection_id2label[len(detection_id2label)] = "no object"
            objects = self.outputs_to_objects(outputs, image.size, detection_id2label, self.detection_class_thresholds)

            # Crop tables
            tables_crops = self.objects_to_crops(image, [], objects, self.detection_class_thresholds, padding=self.crop_padding)

            table_detect_time_end = time.time()
            print(f"Table detection time: {table_detect_time_end - table_detect_time_start:.2f} seconds")

            for table_crop in tables_crops:
                table_crop_time_start = time.time()

                cropped_table = table_crop['image'].convert("RGB")

                # Detect table structure
                pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.structure_model(pixel_values)

                # Get cell objects
                structure_id2label = self.structure_model.config.id2label
                structure_id2label[len(structure_id2label)] = "no object"
                cells = self.outputs_to_objects(outputs, cropped_table.size, structure_id2label, self.structure_class_thresholds)

                # Add padding to cells
                for cell in cells:
                    cell["bbox"] = [cell["bbox"][0]-5, cell["bbox"][1]-5, cell["bbox"][2]+5, cell["bbox"][3]+5]

                # Get cell coordinates
                cell_coordinates = self.get_cell_coordinates_by_col(cells)

                table_crop_time_end = time.time()
                print(f"Table crop time: {table_crop_time_end - table_crop_time_start:.2f} seconds")

                ocr_time_start = time.time()
                # Apply OCR
                data = self.apply_ocr(cell_coordinates, cropped_table)
                
                df = pd.DataFrame.from_dict(data)
                '''
                for x in df.iloc[0]:
                    print(f'x {x} type {type(x)}')

                if not any(str(x).isnumeric() for x in df.iloc[0]):
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)  
                '''
                tables_dataframes.append(df)

                ocr_time_end = time.time()
                print(f"OCR time: {ocr_time_end - ocr_time_start:.2f} seconds")

        return tables_dataframes

# Example usage
if __name__ == "__main__":
    time_start = time.time()
    parser = TableParser()
    tables = parser.parse_pdf_tables('bank_statements/statement_1.pdf')
    for i, table in enumerate(tables, 1):
        print(table)
        print("\n")
    
    time_end = time.time()
    print(f"Execution time: {time_end - time_start:.2f} seconds")
