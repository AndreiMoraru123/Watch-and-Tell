import cv2
from skimage.transform import resize
import os
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from torchvision import transforms
import sys
from YOLO.utils import *
from PIL import Image

cfgfile = "YOLO/yolov3.cfg"
weightfile = "YOLO/yolov3.weights"
namesfile = "YOLO/coco.names"
class_names = load_class_names(namesfile)

sys.path.append('PythonAPI')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

data_loader = get_loader(transform=transform_test,
                         mode='test')

encoder_file = 'encoder-5.ckpt'
decoder_file = 'decoder-5.ckpt'

embed_size = 512
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join('models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('models', decoder_file)))

encoder.to(device)
decoder.to(device)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, colors, classes):
    label = str(classes[class_id])
    color = colors[class_id]
    if label == 'person':
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, str(round(confidence, 2)), (x + 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        roi = img[y:y_plus_h, x:x_plus_w]

        if roi.size != 0:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = resize(roi, (600, 600, 3))
            roi = Image.fromarray((roi * 255).astype(np.uint8))
            roi = transform_test(roi)
            roi = roi.unsqueeze(0)
            roi = roi.to(device)
            features = encoder(roi)
            output = decoder.sample(features.unsqueeze(1))
            sentence = clean_sentence(output)
            cv2.putText(img, sentence, (x - 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def clean_sentence(output):
    sentence = []
    for x in output:
        sentence.append(data_loader.dataset.vocab.idx2word[x])
    sentence = ' '.join(sentence)
    sentence = sentence[:-5]
    sentence = sentence[7:]
    return sentence


def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("window")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # keep video stream open
    while rval:

        width = frame.shape[1]
        height = frame.shape[0]

        with open('YOLO/coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h), colors, classes)

        cv2.imshow("window", frame)

        key = cv2.waitKey(20)
        if key > 0:  # exit by pressing any key
            cv2.destroyAllWindows()
            for i in range(1, 5):
                cv2.waitKey(1)
            return

        # read next frame
        time.sleep(0.05)  # control framerate for computation
        rval, frame = vc.read()


if __name__ == '__main__':
    laptop_camera_go()
