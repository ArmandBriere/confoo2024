# Speech

1. Objective of the presentation
2. Any requirements?
   1. What is my background?
3. Tools we are going to use
4. What is Python
5. What is Golang
6. What is OpenCV
7. What is Yolo
8. Code
9.  Demo

## Objective of the presentation

Golang is not very popular

- Why?

After a few minutes of Google research we can see that big companies are using it

- Uber created their geofence service in Go
- Google created the language and created Kubernetes with it
- Twitch use it to reduce latency
- Netflix use it for high performance proxy

All of those application have a few things in common

- They need speed
- They need to be fast
- They need to scale
- Developers want to have fun

## Any requirements?

Is there any requirements to understand this presentation?

- I expect a small Python and linux knowledge
- I will do my best to explain every technologies
- The slides will be available online after the presentation with all the source code I present here today

## Tools we are going to use

Of course, we will look at the two main programming language

- Python 3.11.7
- Go version go1.22.0 linux/amd64

We will also look at

- OpenCV version:  4.9.0
- [Yolo](https://github.com/pjreddie/darknet)

If we have time we will look into the Cloud integration

- Google Cloud
- Terraform

## Comparison

- get_random_color vs getRandomColor
- load_model vs loadModel
- draw_boxes vs drawBoxes
- draw_bounding_box vs drawBoundingBox
- detect vs detect
- main vs main

New Golang function added

- getOutputLayerNames
- postProcess
- getClassIDAndConfidence
- calculateBoundingBox
- loadYAML

## Things to know

Python is a high-level, general-purpose programming language

Go is a statically typed, compiled high-level programming language designed at Google

How to run them?

```bash
python main.py

go build main.go
./main
```

Faster way to run code in dev

```bash
go run main.go
```

Classic hello world example

```py
print("Hello Confoo!")
```

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello Confoo!")
}
```

- Python is not a typed language
- But you should type it

Let say I am on vaccation or sick and then you need to work on my code, you see this (draw_boxes without typing). What is this doing? How do you know what to provide to this function?

See compare of drawBoxes with and without type

- Typing is not standard and you don't realy need it when you are alone. Programmers are lazy

```py
i = 3

j: int = 3
```

Golang provide `short variable declaration`

- Declaration and typing at the same time

```go
var i int
i = 3

j := 4
```

When playing with OpenCV, you need to read the documentation. You will be looking at the type anyway.


### Functions

```py
def addition(x: int, y: int) -> int:
    return x + y
```

```go
func addition(x int, y int) int {
    return x + y
}
```

### Bonus points


```go
func subtraction(x int, y int) int {
    var z int
    return x - y
}
```

```bash
# confoo24.inference/playground
./playground.go:23:6: z declared and not used
```


```go
func subtraction(x int, y int) (z int) {
    z = x - y
    return
}
```

## Let's dive in

### main vs main

```py
if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=str("person.jpg"), help="Path to input image.")
    args = parser.parse_args()

    # Load the model and classes
    net_model: cv2.dnn.DetectionModel = load_model(
        "yolov4-tiny.weights", "yolov4-tiny.cfg"
    )
    with open("classes.yml", "r") as file:
        config: dict = yaml.safe_load(file)
        classes: List[str] = config["names"]

    detect(net_model, args.img, classes)
```

```go
// main function
func main() {
    // Parse the command line arguments
    img := flag.String("img", "person.jpg", "Path to input image.")
    flag.Parse()

    // Load the model and classes
    netModel := loadModel("yolov4-tiny.weights", "yolov4-tiny.cfg")
    classes := loadYAML("classes.yml")

    detect(netModel, *img, classes)
}
```

- Do I know the type of `img`?
  - `var img *string`

Oh that's a pointer...

Let's go simple, a pointer is just here to save memory space. If you have a pointer, you can use the value directly of the address of the value

Under the hood, Python is using it:

- Object ref
- Pointer to the object type

A list in python is a pointer to the content

- Insert meme

#### Pointer example

```go
func main() {
    x := 42
    p := &x

    fmt.Println("x", x)
    fmt.Println("p", p)
    fmt.Println("&x", &x)
    fmt.Println("*p", *p)
    fmt.Println("*&x", *&x)
    fmt.Println("*&*&*&*&*&*&*&*&*&x", *&*&*&*&*&*&*&*&*&x)
}
```

```bash
x   42
p   0xc000012158
&x  0xc000012158
*p  42
*&x 42
*&*&*&*&*&*&*&*&*&x 42
```

### Back to main

- `*img` is just the file name

```bash
python main.py --img bus.jpg

go run main.go --img bus.jpg
```

What do we need in my program then?

- `net_model`
  - This is the AI model
- `config`
  - This is the config files
- `classes`
  - It seems to be a list of classes names

Pretty simple here


### Load model

```py
def load_model(weights_file: str, cfg_file: str) -> cv2.dnn.DetectionModel:
    """Load yolo model with cv.dnn."""

    net: cv2.dnn.Net = cv2.dnn.readNet(weights_file, cfg_file)

    if net.empty():
        raise ValueError(f"Model {weights_file} and {cfg_file} not loaded")

    model: cv2.dnn.DetectionModel = cv2.dnn.DetectionModel(net)

    return model
```


```go
// init loads the Yolo model.
func loadModel(yoloModelName string, yoloModelCfg string) gocv.Net {
    net := gocv.ReadNet(yoloModelName, yoloModelCfg)

    if net.Empty() {
        panic("Failed to load the model")
    }

    return net
}
```

- Call `readNet` function to upload weight and config
- Setup input params in Python but not in Golang
- Return the model


What is `cv2.dnn.readNet`?

- `cv2` is OpenCV
- `dnn` is the deep neural network module
- `readNet` does "Read deep learning network represented in one of the supported formats."

Neural network need to be supported, what's the list?

See opencv readnet screenshot

OpenCV is reading the model config and create the necessary layer for you in the background. Do you need to understand AI to do this? No

### Error handling in Go

Go error handling sounds strange at first

```go
func httpGet(url string) (string, bool) {
    resp, err := http.Get(url)
    if err != nil {
        return "", false
    }
    defer resp.Body.Close()

    bodyBytes, err := io.ReadAll(resp.Body)
    if err != nil {
        log.Fatal(err)
        // or panic(err)
    }
    bodyString := string(bodyBytes)
    return bodyString, true
}
```

`err` in Go in a value.

Common error handling in Go: Return an error as a value so the caller can check for it.

## What is this yolo model?

You only look once (YOLO)

It's a merge between

- Image Recognition
- Image Localization

What do I see on that image?

Where is it on the image?

The idea of YoloV4 is to find a balance between Speed and accuracy

### Darknet

Open source YoloV4

Darknet is an open source network framework written in C and CUDA.

Supports CPU and GPU

### COCO

COCO is a large-scale object detection, segmentation, and captioning dataset.

## Load yaml

```py
with open("classes.yml", "r") as file:
    config: dict = yaml.safe_load(file)
    classes: List[str] = config["names"]
```

```go
// loadYAML loads the YAML file and returns the Config struct
func loadYAML(filename string) []string {
    var config Config

    data, err := os.ReadFile(filename)
    if err != nil {
        panic(err)
    }

    err = yaml.Unmarshal(data, &config)
    if err != nil {
        panic(err)
    }

    return config.Names
}
```

This is a new line structure:

- `yaml.Unmarshal(data, &config)`
- Unmarshal the data from the file and write it at the address of `config`

> Marshalling is the process of transforming memory representation of an object into a data format suitable for storage

- [Wikipedia](https://en.wikipedia.org/wiki/Marshalling_(computer_science))

### Save point

What have we done until here?

- Basic Python and Go structure
  - How to run program
  - Variables, function
  - Standard file structure
- Typing
- Pointers
- Error handling
- Basic Yolo concept
- File read and data marshalling

## Detect method

```py
def detect(net: cv2.dnn.DetectionModel, input_image: str, classes: List[str]):
    """Perform inference on the input image."""

    # Read the image
    original_image: np.ndarray = cv2.imread(input_image)

    now: datetime = datetime.now()

    # Perform inference
    detected_classes, scores, boxes = net.detect(original_image, THRESHOLD, 0.25)

    elapsed_time: float = (datetime.now() - now).total_seconds() * 1000
    print(f"{elapsed_time:.0f}")

    draw_boxes(original_image, boxes, detected_classes, classes, scores)

    # Write the image to disk
    cv2.imwrite("python_inference.jpg", original_image)
```

1. Read the image
2. Use the network to detect what's on the image
   1. Classes
   2. Scores (accuracy)
   3. Bounding boxes
3. Draw boxes on the image
4. Write the image to disk

Bonus: We track the elapsed time

```go
// detect perform inference on the input image
func detect(net gocv.Net, imgPath string, classes []string) {

    // Read the image
    b, err := os.ReadFile(imgPath)
    if err != nil {
        panic(err)
    }

    img, err := gocv.IMDecode(b, gocv.IMReadAnyColor)
    if err != nil {
        panic(err)
    }
    defer img.Close()

    // Preprocess the image and prepare blob for model
    blob := gocv.BlobFromImage(img, 1/255.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
    net.SetInput(blob, "")

    startingTime := time.Now()

    probs := net.ForwardLayers(getOutputLayerNames(&net))

    // Clear gocv.Mat
    defer func() {
        for _, prob := range probs {
            prob.Close()
        }
    }()

    // Postprocess the output
    detected := postProcess(probs, 0.4, float32(img.Cols()), float32(img.Rows()), classes)

    inferenceTime := time.Since(startingTime)
    fmt.Println(inferenceTime.Milliseconds())

    if len(detected) == 0 {
        fmt.Println("No classes found")
    } else {
        drawBoxes(&img, detected)
    }

    // Save the image
    gocv.IMWrite("golang_inference.jpg", img)
}
```

1. Read file
2. Decode the content to img
3. Convert to img to blob
4. Use the network to ForwardLayers
   1. Forward until `getOutputLayerNames(&net)`
5. Apply post process to transform output
6. Draw boxes on the image
7. Write the image to disk


```go
// detect perform inference on the input image
func detect(net gocv.Net, imgPath string, classes []string) {

    // Read the image
    b, err := os.ReadFile(imgPath)

    img, err := gocv.IMDecode(b, gocv.IMReadAnyColor)

    // Preprocess the image and prepare blob for model
    blob := gocv.BlobFromImage(img, 1/255.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
    net.SetInput(blob, "")

    // Perform inference
    probs := net.ForwardLayers(getOutputLayerNames(&net))

    // Postprocess the output
    detected := postProcess(probs, 0.4, float32(img.Cols()), float32(img.Rows()), classes)

    // Draw boxes on image
    drawBoxes(&img, detected)

    // Save the image
    gocv.IMWrite("golang_inference.jpg", img)
}
```

### Reading a file

```py
# Read the image
original_image: np.ndarray = cv2.imread(input_image)
```

```go
// Read the image
b, err := os.ReadFile(imgPath)
if err != nil {
    panic(err)
}

img, err := gocv.IMDecode(b, gocv.IMReadAnyColor)
if err != nil {
    panic(err)
}
defer img.Close()
```

- What is `defer`?
  - Do it later after the execution of this function

In go, we transition from a path, to a `var v []byte` to a `gocv.Mat` (multidimensional array)

### Using the network

```py
# Perform inference
net.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
detected_classes, scores, boxes = net.detect(original_image, THRESHOLD, 0.25)
```

```go
// Preprocess the image and prepare blob for model
blob := gocv.BlobFromImage(img, 1/255.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
net.SetInput(blob, "")

// Perform inference
probs := net.ForwardLayers(getOutputLayerNames(&net))

// Postprocess the output
detected := postProcess(probs, 0.4, float32(img.Cols()), float32(img.Rows()), classes)
```

Define model input:

```py
net.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
```

```go
gocv.BlobFromImage(img, 1/255.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
```

Image, Scale, Size, Mean value to substract for channels, swapRB, crop

Detect objects

```py
detected_classes, scores, boxes = net.detect(original_image, THRESHOLD, 0.25)
```

```go
// Perform inference
probs := net.ForwardLayers(getOutputLayerNames(&net))

// Postprocess the output
detected := postProcess(probs, 0.4, float32(img.Cols()), float32(img.Rows()), classes)
```

![layers](./assets/yolov4-tiny.cfg.png)


- Background image source https://medium.com/swlh/python-vs-golang-select-the-best-one-to-level-up-your-business-1a6d0fb32991
- OpenCV https://opencv.org/
- GoCV https://gocv.io/
- OpenCV-Python https://github.com/opencv/opencv-python
- Google trends https://trends.google.com/trends/explore?date=today%205-y&q=%2Fm%2F09gbxjr,%2Fm%2F05z1_
- Uber using Golang https://www.uber.com/en-CA/blog/go-geofence-highest-query-per-second-service/
- Twitch using Golang https://medium.com/twitch-news/gos-march-to-low-latency-gc-a6fa96f06eb7
- Netflix high performance proxy in go https://netflixtechblog.com/application-data-caching-using-ssds-5bf25df851ef
- Netron app https://netron.app/
- Geeksforgeeks https://www.geeksforgeeks.org/go-pointer-to-pointer-double-pointer/