using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cubeRender : MonoBehaviour
{
    [HideInInspector] public Transform pointA;
    public Transform pointB;
    private GameObject cubeObject;
    private MeshRenderer cubeRenderer;
    private Transform cubeTransform;

    [Header("Cube Settings")]
    public Material cubeMaterial;
    public Color cubeColor = Color.black;
    public Vector3 cubeScale = new Vector3(0.1f, 0.1f, 1f); // Width, Height, Length

    void SetUpCube()
    {
        pointA = this.transform;

        // Create a new GameObject for the cube
        cubeObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cubeObject.name = "ConnectingCube";
        cubeObject.transform.SetParent(this.transform);

        cubeTransform = cubeObject.transform;
        cubeRenderer = cubeObject.GetComponent<MeshRenderer>();

        // Set up material
        if (cubeMaterial != null)
        {
            cubeRenderer.material = cubeMaterial;
        }
        else
        {
            cubeRenderer.material = new Material(Shader.Find("Standard"));
            cubeRenderer.material.color = cubeColor;
        }
    }

    void Start()
    {
        try
        {
            SetUpCube();
        }
        catch
        {
            Debug.LogError("Failed to set up cube");
        }
    }

    void Update()
    {
        try
        {
            if (pointB != null && cubeObject != null)
            {
                // Calculate position (midpoint between the two points)
                Vector3 midPoint = (pointA.position + pointB.position) / 2f;
                cubeTransform.position = midPoint;

                // Calculate rotation to face from pointA to pointB
                Vector3 direction = pointB.position - pointA.position;
                if (direction != Vector3.zero)
                {
                    cubeTransform.rotation = Quaternion.LookRotation(direction);
                }

                // Calculate scale based on distance between points
                float distance = Vector3.Distance(pointA.position, pointB.position);
                Vector3 newScale = cubeScale;
                newScale.z = distance*cubeScale.z; // Stretch the cube along its length
                cubeTransform.localScale = newScale;
            }
        }
        catch
        {
            Debug.LogError("Error updating cube position/rotation");
        }
    }

    // Optional: Method to change cube color at runtime
    public void SetCubeColor(Color newColor)
    {
        if (cubeRenderer != null)
        {
            cubeRenderer.material.color = newColor;
        }
    }

    // Optional: Method to change cube material at runtime
    public void SetCubeMaterial(Material newMaterial)
    {
        if (cubeRenderer != null && newMaterial != null)
        {
            cubeRenderer.material = newMaterial;
        }
    }

    void OnDestroy()
    {
        // Clean up the cube when this component is destroyed
        if (cubeObject != null)
        {
            DestroyImmediate(cubeObject);
        }
    }
}
