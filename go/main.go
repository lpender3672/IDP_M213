package main

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/template/html"

	"github.com/hybridgroup/mjpeg"
	"gocv.io/x/gocv"
)


var (
	// deviceID int
	err      error
	webcam   *gocv.VideoCapture
	stream   *mjpeg.Stream
)

func main() {

	// open webcam
	webcam, err := gocv.OpenVideoCapture("http://localhost:8081/stream/video.mjpeg")
	if err != nil {
		fmt.Printf("Error opening capture device")
		return
	}
	defer webcam.Close()

	// create the mjpeg stream
	stream = mjpeg.NewStream()

	// start capturing
	go mjpegCapture()

	engine := html.New("./templates", ".html")

    app := fiber.New(fiber.Config{
        Views: engine, //set as render engine
    })
    app.Static("/public", "./public")
    app.Get("/", mainPage)
	app.Get("/stream", stream)
    app.Listen(":3000")

	
}

func mainPage(c *fiber.Ctx) error {
    return c.Render("mainpage", nil)
}

// // This route path will match requests to the root route, "/":
// func streamer(c *fiber.Ctx) error {
// 	return c.SendStream(stream)
// }

func mjpegCapture() {
	img := gocv.NewMat()
	defer img.Close()

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed")
			return
		}
		if img.Empty() {
			continue
		}

		buf, _ := gocv.IMEncode(".jpg", img)
		stream.UpdateJPEG(buf.GetBytes())
		buf.Close()
	}
}