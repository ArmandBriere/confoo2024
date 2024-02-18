package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
)

func main() {

	x := 42
	p := &x

	fmt.Println("x", x)
	fmt.Println("p", p)
	fmt.Println("&x", &x)
	fmt.Println("*p", *p)
	fmt.Println("*&x", *&x)
	fmt.Println("*&*&*&*&*&*&*&*&*&x", *&*&*&*&*&*&*&*&*&x)

	fmt.Println("addition(2, 3)", addition(2, 3))
	fmt.Println("subtraction(2, 3)", subtraction(2, 3))

	httpGetResult, ok := httpGet("https://www.google.com")

	if !ok {
		fmt.Println("Failed to get the URL")
		return
	}

	fmt.Println("httpGetResult", httpGetResult)
}

func addition(x int, y int) int {
	return x + y
}

func subtraction(x int, y int) (z int) {
	z = x - y
	return
}

func httpGet(url string) (string, bool) {
	resp, err := http.Get(url)
	if err != nil {
		return "", false
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	bodyString := string(bodyBytes)
	return bodyString, true
}
