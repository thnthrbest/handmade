using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.ComponentModel;

public class HandTracking : MonoBehaviour
{
    // Start is called before the first frame update
    public UDPReceive udpReceive;

    [TextArea(20, 50)]
    public string data;
    public GameObject[][] handPoints = new GameObject[2][];
    public GameObject[] handParents = new GameObject[2];

    void Start()
    {
        timeLeft = updateInterval;
        for (int i = 0; i < handParents.Length; i++)
        {
            GameObject[] temp = new GameObject[23];

            for (int n = 0; n < handParents[i].transform.childCount; n++)
            {
                temp[n] = handParents[i].transform.GetChild(n).gameObject;
            }
            handPoints[i] = temp;
        }

    }

    // Update is called once per frame

    public float updateInterval = 0.5f; // The interval at which to update the FPS display
    private float accum = 0f; // FPS accumulated over the interval
    private int frames = 0; // Frames drawn over the interval
    private float timeLeft; // Time left for current interval
    public float shift = 7;
    public string[] points;
    [TextArea(20, 50)]

    public string[] hands;
    public string[][] handsPoints = new string[2][];
    public string[] handPoint1;
    public string[] handPoint2;




    void Update()
    {
        try
        {
            data = udpReceive.data;
            hands = data.Split("_");
          
            // Update 2 Hands
            for (int i = 0; i < hands.Length; i++)
            {
                string temp = hands[i];
                if (temp.Length >= 2)
                {
                    temp = temp.Remove(temp.Length - 1, 1); // Remove last char
                    temp = temp.Remove(0, 1);               // Remove first char
                    hands[i] = temp;
                    handsPoints[i] = temp.Split(",");
                }
                else
                {
                    hands[i] = ""; // Or some fallback
                }

            }
            handPoint1 = handsPoints[0];
            handPoint2 = handsPoints[1];


            //0        1*3      2*3
            //x1,y1,z1,x2,y2,z2,x3,y3,z3

            for (int i = 0; i < handsPoints.Length; i++)
            {
                if (handsPoints[i] != null && handsPoints[i].Length >= 63) // 21 * 3 = 63
                {
                    for (int j = 0; j < 21; j++)
                    {
                        try
                        {
                            float x = 32.83f - float.Parse(handsPoints[i][j * 3]) / 100f;
                            float y = float.Parse(handsPoints[i][j * 3 + 1]) / 100f;
                            float z = float.Parse(handsPoints[i][j * 3 + 2]) / 100f;

                            if (handPoints[i] != null && handPoints[i].Length > j && handPoints[i][j] != null)
                            {

                                handPoints[i][j].transform.localPosition = new Vector3(x, y, z);
                              

                            }
                            else
                            {
                                Debug.LogWarning($"handPoints[{i}][{j}] is null or out of range");
                            }
                        }
                        catch (Exception e)
                        {
                            Debug.LogError($"Error parsing data at handsPoints[{i}]: {e.Message}");
                        }
                    }
                }
                else
                {
                    Debug.LogWarning($"handsPoints[{i}] has insufficient data: {handsPoints[i]?.Length}");
                }

                
            }

        }
        catch (Exception err)
        {
            Debug.Log(err.ToString());
        }



    }


   


}