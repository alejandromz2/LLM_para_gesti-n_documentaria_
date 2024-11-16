const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 3080;

// Middleware
app.use(bodyParser.json()); // O puedes usar app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));

// Ruta POST
app.post('/', async (req, res) => {
  const { message } = req.body; // Recibe el mensaje desde el cliente
  console.log('POST message received:', message);

  try {
    // Hacer el POST a http://localhost:3001/chat usando el mensaje recibido
    const response = await axios.post('http://localhost:3001/chat', {
      question: message
    });

    console.log('Response from 3001:', response.data);
    
    res.json({
      message: response.data.respuesta // Enviar la respuesta de localhost:3001 al cliente
    });

  } catch (error) {
    console.error('Error in POST to localhost:3001:', error.message);
    if (error.response){
      console.error('Response data', error.response.data)
      console.error("Status: ", error.response.status)

    }
    else if (error.request) {
      console.error("No response received", error.request)
    }
    res.status(500).json({ error: 'Error posting to chat service' });
  }
});

// Arrancar el servidor
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
