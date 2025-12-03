using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshRenderer), typeof(MeshFilter))]
public class customHandShape : MonoBehaviour
{
    public Transform[] handPoints = new Transform[7]; // 7 points
    public float extrusionDepth = -0.08f;
    // public Shader shader;
    // public Material shapeMaterial;

    private Mesh mesh;

    void Start()
    {
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;
        MeshRenderer mr = GetComponent<MeshRenderer>();
        mr.material = new Material(Shader.Find("Custom/fillHandShader"));
        mr.material.color = Color.white; // Optional color
        // if (shapeMaterial != null)
        //     GetComponent<MeshRenderer>().material = shapeMaterial;

        GenerateExtrudedMesh();
    }

    void Update()
    {
        GenerateExtrudedMesh(); // Call in Update for real-time motion
    }

    void GenerateExtrudedMesh()
    {
        if (handPoints.Length < 3) return;

        // Get front and back vertices
        Vector3[] frontVerts = new Vector3[handPoints.Length];
        Vector3[] backVerts = new Vector3[handPoints.Length];

        for (int i = 0; i < handPoints.Length; i++)
        {
            try
            {
                Vector3 local = transform.InverseTransformPoint(handPoints[i].position);
                frontVerts[i] = local;
                backVerts[i] = local + Vector3.back * extrusionDepth; // Extrude in -Z
            }
            catch
            {
                
            }
        }

        // Combine vertices
        List<Vector3> vertices = new List<Vector3>();
        vertices.AddRange(frontVerts);
        vertices.AddRange(backVerts); // Now total vertices = 14 (7 front + 7 back)

        List<int> triangles = new List<int>();

        // Triangulate front face (0–6)
        triangles.AddRange(Triangulate(frontVerts, 0));

        // Triangulate back face (7–13) (reverse order to flip normal)
        triangles.AddRange(Triangulate(backVerts, handPoints.Length, true));

        // Connect sides
        for (int i = 0; i < handPoints.Length; i++)
        {
            int next = (i + 1) % handPoints.Length;

            int f0 = i;
            int f1 = next;
            int b0 = i + handPoints.Length;
            int b1 = next + handPoints.Length;

            // Two triangles per quad (f0–f1–b1, f0–b1–b0)
            triangles.Add(f0);
            triangles.Add(f1);
            triangles.Add(b1);

            triangles.Add(f0);
            triangles.Add(b1);
            triangles.Add(b0);
        }

        mesh.Clear();
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();  // For culling and visibility

    }

    List<int> Triangulate(Vector3[] verts, int offset, bool reverse = false)
    {
        List<int> tri = new List<int>();
        for (int i = 1; i < verts.Length - 1; i++)
        {
            if (!reverse)
            {
                tri.Add(offset + 0);
                tri.Add(offset + i);
                tri.Add(offset + i + 1);
            }
            else
            {
                tri.Add(offset + 0);
                tri.Add(offset + i + 1);
                tri.Add(offset + i);
            }
        }
        return tri;
    }
}
