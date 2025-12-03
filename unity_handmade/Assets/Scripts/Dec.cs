using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.IO;
using System;

public class Dec : MonoBehaviour
{
    public RawImage rawImage;   
    private WebCamTexture webcamTexture;

    public TcpClient client;
    public NetworkStream stream;
    public BinaryWriter writer;

    [Header("Script")]
    public string serverAddress = "127.0.0.1";
    public int serverPort = 5055;

    Texture2D frame;

    void Start()
    {
        try
        {
            // Start webcam
            WebCamDevice[] devices = WebCamTexture.devices;
            try
            {
                webcamTexture = new WebCamTexture(devices[1].name);
            }
            catch (Exception e)
            {
                Debug.LogError(e);
                webcamTexture = new WebCamTexture();
            }

            rawImage.texture = webcamTexture;
            webcamTexture.Play();

            frame = new Texture2D(webcamTexture.width, webcamTexture.height);

            // Connect to Python
            client = new TcpClient(serverAddress, serverPort);
            stream = client.GetStream();
            writer = new BinaryWriter(stream);

        }
        catch (Exception e)
        {
            Debug.LogError(e);
        }
    }

    void Update()
    {
        if (!webcamTexture.isPlaying) return;

        // copy webcam frame → Texture2D
        frame.SetPixels(webcamTexture.GetPixels());
        frame.Apply();

        // encode JPG
        byte[] bytes = frame.EncodeToJPG();

        // send to python
        if (writer != null)
        {
            writer.Write(bytes.Length);
            writer.Write(bytes);
            writer.Flush();
        }
    }

    void OnApplicationQuit()
    {
        if (writer != null) writer.Close();
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }

    void OnDestroy()
    {
        if (writer != null) writer.Close();
        if (stream != null) stream.Close();
        if (client != null) client.Close();
        if (webcamTexture != null) webcamTexture.Stop();
    }
}
