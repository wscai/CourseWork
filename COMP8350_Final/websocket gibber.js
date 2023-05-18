const socket = new WebSocket("ws://localhost:13001")
socket.addEventListener("open", (event) => {
  console.log("connected to websocket")
})

socket.addEventListener("message", (event) => {
    console.log(event.data)
})