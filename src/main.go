package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gocv.io/x/gocv"
	"gopkg.in/yaml.v2"
)

var threshold float32 = 0.5

// getOutputLayerNames outputs layer names from the Yolo model.
func getOutputLayerNames(net *gocv.Net) []string {
	var outputLayers []string
	for _, i := range net.GetUnconnectedOutLayers() {
		layer := net.GetLayer(i)
		layerName := layer.GetName()
		outputLayers = append(outputLayers, layerName)
	}

	return outputLayers
}

// getRandomColor returns a random color.
func getRandomColor() color.RGBA {
	r := uint8(rand.Intn(256))
	g := uint8(rand.Intn(256))
	b := uint8(rand.Intn(256))
	return color.RGBA{r, g, b, 0}
}

// init loads the Yolo model.
func loadModel(yoloModelName string, yoloModelCfg string) gocv.Net {
	net := gocv.ReadNet(yoloModelName, yoloModelCfg)
	if net.Empty() {
		// fmt.Println("Reading network model from file: empty model")
		panic(1)
	}

	return net
}

// drawBoxes draws multiple boxes on the image.
func drawBoxes(originalImage *gocv.Mat, detectedObjects []*DetectedObject) {
	for _, detectedObject := range detectedObjects {
		drawBoundingBox(originalImage, detectedObject.Rect, detectedObject.ClassName, detectedObject.Confidence)
	}
}

// drawBoundingBox draws a box on the image.
func drawBoundingBox(img *gocv.Mat, rect image.Rectangle, className string, confidence float32) {
	// Get a random color
	randomColor := getRandomColor()

	// Prepare label
	confidenceStr := strconv.FormatFloat(float64(confidence), 'f', 2, 32)
	text := className + " (" + confidenceStr + ")"

	// Draw rectangle and text
	gocv.Rectangle(img, rect, randomColor, 2)
	gocv.PutText(img, text, image.Pt(rect.Min.X-10, rect.Min.Y-10), gocv.FontHersheySimplex, 0.8, randomColor, 2)
}

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

// DetectedObject Store detected object info
type DetectedObject struct {
	Rect       image.Rectangle
	ClassID    int
	ClassName  string
	Confidence float32
}

// postProcess processes the output of the Yolo model.
func postProcess(detections []gocv.Mat, nmsThreshold float32, frameWidth, frameHeight float32, netClasses []string) []*DetectedObject {
	var detectedObjects []*DetectedObject
	var boundingBoxes []image.Rectangle
	var confidences []float32

	for i, yoloLayer := range detections {
		cols := yoloLayer.Cols()
		data, err := detections[i].DataPtrFloat32()
		if err != nil {
			panic(err)
		}

		// Iterate Yolo results
		for j := 0; j < yoloLayer.Total(); j += cols {
			row := data[j : j+cols]
			scores := row[5:]
			classID, confidence := getClassIDAndConfidence(scores)
			className := netClasses[classID]

			// Remove the bounding boxes with low confidence
			if confidence > threshold {
				// Calculate bounding box
				boundingBox := calculateBoundingBox(frameWidth, frameHeight, row)

				// Append data to the lists
				confidences = append(confidences, confidence)
				boundingBoxes = append(boundingBoxes, boundingBox)
				detectedObjects = append(detectedObjects, &DetectedObject{
					Rect:       boundingBox,
					ClassName:  className,
					ClassID:    classID,
					Confidence: confidence,
				})
			}
		}
	}

	if len(boundingBoxes) == 0 {
		return nil
	}
	indices := make([]int, len(boundingBoxes))
	for i := range indices {
		indices[i] = -1
	}

	// Apply non-maximum suppression
	indices = gocv.NMSBoxes(boundingBoxes, confidences, threshold, nmsThreshold)
	filteredDetectedObjects := make([]*DetectedObject, 0, len(detectedObjects))
	for i, idx := range indices {
		if idx < 0 || (i != 0 && idx == 0) {
			// Eliminate zeros, since they are filtered by NMS (except first element)
			// Also filter all '-1' which are undefined by default
			continue
		}
		filteredDetectedObjects = append(filteredDetectedObjects, detectedObjects[idx])
	}

	return filteredDetectedObjects
}

// getClassIDAndConfidence returns the class ID and confidence of the detected object.
func getClassIDAndConfidence(x []float32) (int, float32) {
	classID := 0
	confidence := float32(0.0)
	for i, y := range x {
		if y > confidence {
			confidence = y
			classID = i
		}
	}

	return classID, confidence
}

// calculateBoundingBox calculates the bounding box of the detected object.
func calculateBoundingBox(frameWidth, frameHeight float32, row []float32) image.Rectangle {
	if len(row) < 4 {
		return image.Rect(0, 0, 0, 0)
	}
	centerX := int(row[0] * frameWidth)
	centerY := int(row[1] * frameHeight)
	width := int(row[2] * frameWidth)
	height := int(row[3] * frameHeight)
	left := centerX - width/2
	top := centerY - height/2

	return image.Rect(left, top, left+width, top+height)
}

// Config struct use to extract the class names from the YAML file
type Config struct {
	Names []string `yaml:"names"`
}

// loadYAML loads the YAML file and returns the Config struct
func loadYAML(filename string) Config {
	var config Config

	data, err := os.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	err = yaml.Unmarshal(data, &config)
	if err != nil {
		panic(err)
	}

	return config
}

// main function
func main() {
	// Parse the command line arguments
	img := flag.String("img", "person.jpg", "Path to input image.")
	flag.Parse()

	// Load the model and classes
	netModel := loadModel("yolov4-tiny.weights", "yolov4-tiny.cfg")
	config := loadYAML("classes.yml")
	classes := config.Names

	detect(netModel, *img, classes)
}
