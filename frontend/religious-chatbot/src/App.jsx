import { useState } from "react";
import LandingPage from "./Landingpage";
import Chatbot from "./Chatbot";

export default function App() {
  const [screen, setScreen]               = useState("landing"); // "landing" | "chat"
  const [selectedReligion, setSelectedReligion] = useState(null);

  const handleEnterChat = (religion) => {
    setSelectedReligion(religion);
    setScreen("chat");
  };

  const handleSwitchReligion = () => {
    setScreen("landing");
    setSelectedReligion(null);
  };

  if (screen === "chat" && selectedReligion) {
    return (
      <Chatbot
        religion={selectedReligion}
        onSwitchReligion={handleSwitchReligion}
      />
    );
  }

  return <LandingPage onEnterChat={handleEnterChat} />;
}