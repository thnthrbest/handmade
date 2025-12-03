using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LineManager : MonoBehaviour
{
    public List<GameObject> point;
    private int[] listOfLineConnect = { 17, 0, 1, 2, 3, 0, 5, 6, 7, 5, 9, 10, 11, 9, 13, 14, 15, 13, 17, 18, 19 };
    private int[] listOfFillHandLine = { 0, 1, 2, 5, 9, 13, 17 };
    private void Start()
    {
        for (int i = 0; i < this.transform.childCount; i++)
        {
            point.Add(this.transform.GetChild(i).gameObject);

        }
        for (int i = 0; i < point.Count; i++)
        {
            try
            {
                point[i].GetComponent<cubeRender>().pointB = point[listOfLineConnect[i]].transform;

            }
            catch
            {

            }
        }

        this.GetComponent<customHandShape>().handPoints = new Transform[7];
        for (int i = 0; i < listOfFillHandLine.Length; i++)
        {
            try
            {
                this.GetComponent<customHandShape>().handPoints[i] = this.transform.GetChild(listOfFillHandLine[i]).transform;
            }
            catch
            {

            }
        }

    }

}
