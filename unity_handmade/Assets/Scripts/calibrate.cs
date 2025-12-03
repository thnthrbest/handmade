using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;


public class calibrate : MonoBehaviour
{
    [Header("Script")]
    [SerializeField] private UDPReceive uDPReceive;
    [Header("GameObject UI")]
    [SerializeField] private TMP_Text StateText;
    private int State = 1;

    private int[] value = new int[2];


    public void StartCalibrate()
    {
        if (State == 1)
        {
            value[0] = Int32.Parse(uDPReceive.data);
            StateText.text = "กรุณาหันมือ";
        }
        else if (State == 2)
        {
            value[1] = Int32.Parse(uDPReceive.data);
            StateText.text = "การตั้งค่ามือเสร็จแล้ว";
            Debug.LogWarning((value[0] + value[1]) / 2);
            PlayerPrefs.SetInt("threshold", (value[0] + value[1]) / 2);
            State = 1;

        }
        State++;

    }

}
