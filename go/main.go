package main

import (
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/template/html"
	"gocv.io/x/gocv"
)

func main() {

	engine := html.New("./templates", ".html")

    app := fiber.New(fiber.Config{
        Views: engine, //set as render engine
    })
    app.Static("/public", "./public")
    app.Get("/", mainPage)
    app.Listen(":3000")

	webcam, _ := gocv.OpenVideoCapture("http://localhost:8081/stream/video.mjpeg")
	window := gocv.NewWindow("Hello")
	img := gocv.NewMat()

	for {
		webcam.Read(&img)
		window.IMShow(img)
		window.WaitKey(1)
	}
}

func mainPage(c *fiber.Ctx) error {
    return c.Render("mainpage", nil)
}
