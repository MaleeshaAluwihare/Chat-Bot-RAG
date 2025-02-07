import React, { useState } from "react";

export default function ChatbotUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);

  // Handles file upload to the backend
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("File uploaded:", data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  // Handles sending chat messages to the backend
  const handleSend = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });
      const data = await response.json();
      setMessages([...newMessages, { role: "bot", content: data.answer }]);
    } catch (error) {
      console.error("Error sending message:", error);
    }
    setInput("");
  };

  return (
    <div style={{ width: "500px", margin: "0 auto", padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center" }}>Chatbot with RAG</h2>

      {/* PDF File Upload */}
      <div style={{ marginBottom: "10px", textAlign: "center" }}>
        <input type="file" accept="application/pdf" onChange={handleFileUpload} />
      </div>

      {/* Chat Messages Display */}
      <div
        style={{
          border: "1px solid #ccc",
          height: "400px",
          padding: "10px",
          overflowY: "scroll",
          marginBottom: "10px",
          backgroundColor: "#f9f9f9",
        }}
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            style={{
              textAlign: msg.role === "user" ? "right" : "left",
              marginBottom: "10px",
            }}
          >
            <span
              style={{
                display: "inline-block",
                padding: "8px 12px",
                backgroundColor: msg.role === "user" ? "#daf8cb" : "#e0e0e0",
                borderRadius: "8px",
                maxWidth: "70%",
              }}
            >
              {msg.content}
            </span>
          </div>
        ))}
      </div>

      {/* Input Area and Send Button */}
      <div style={{ display: "flex" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          style={{
            flex: "1",
            padding: "10px",
            border: "1px solid #ccc",
            borderRadius: "4px",
            fontSize: "14px",
          }}
        />
        <button
          onClick={handleSend}
          style={{
            padding: "10px 15px",
            marginLeft: "10px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "14px",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
