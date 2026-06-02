import Header from "./components/Header";
import MainContent from "./components/MainContent";
import Footer from "./components/Footer";
import "bootstrap/dist/css/bootstrap.min.css";
import "./styles/layout.css";
import "./styles/chat.css";
import "./styles/sources.css";

function App() {
  return (
    <div>
      <Header />
      <MainContent />
      <Footer />
    </div>
  );
}

export default App;
