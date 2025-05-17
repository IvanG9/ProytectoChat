"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

type Message = {
  text: string;
  sender: "user" | "bot";
  model?: string;
};

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("RNN"); // Modelo por defecto

  const handleSend = async (): Promise<void> => {
    if (input.trim() === "") return;

    // Agregar mensaje del usuario
    setMessages((prev) => [
      ...prev,
      { text: input, sender: "user", model: selectedModel },
    ]);
    setInput("");

    try {
      const response = await fetch("http://127.0.0.1:5000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, model: selectedModel }),
      });

      if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);

      const data = await response.json();

      let responseText = "";

      if (selectedModel === "ST" && data.sentiment) {
        responseText = `Sentimiento detectado: ${data.sentiment}`;
      } else if (selectedModel === "RNN" && data.generated_text) {
        responseText = data.generated_text;
      } else if (selectedModel === "LLM" && data.generated_text) {
        responseText = data.generated_text;
      }
      else {
        responseText = "Respuesta no reconocida.";
      }

      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          { text: responseText, sender: "bot", model: selectedModel },
        ]);
      }, 500);
    } catch (error) {
      console.error("Error al enviar datos:", error);
      setMessages((prev) => [
        ...prev,
        {
          text: "Hubo un error al procesar la solicitud.",
          sender: "bot",
          model: selectedModel,
        },
      ]);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-4 h-screen flex flex-col">
      <header className="text-center">
        <h1 className="text-2xl font-bold">Chat Inteligente</h1>
        <p className="text-gray-600">
          Interactúa con el modelo y obtén respuestas en tiempo real.
        </p>
      </header>

      {/* Selector de modelo */}
      <div>
        <label
          htmlFor="model-select"
          className="block text-sm font-medium text-gray-700"
        >
          Selecciona un modelo:
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="mt-1 p-2 border rounded-md w-full bg-white"
        >
          <option value="RNN">RNN (Generador)</option>
          <option value="ST">ST (Sentimientos)</option>
          <option value="LLM">LLM (Generador Noticias)</option>
        </select>
      </div>

      {/* Contenedor de mensajes */}
      <div className="flex-1 border rounded-lg p-4 space-y-2 overflow-y-auto bg-gray-50">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-2 rounded-xl max-w-[75%] ${
                msg.sender === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-200 text-black"
              }`}
            >
              <p className="text-sm">
                <strong>{msg.sender === "user" ? "Tú" : msg.model}:</strong>{" "}
                {msg.text}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Input y botón */}
      <div className="flex space-x-2 mt-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Escribe tu mensaje..."
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleSend();
            }
          }}
          className="flex-1"
        />
        <Button onClick={handleSend}>Enviar</Button>
      </div>
    </div>
  );
}
