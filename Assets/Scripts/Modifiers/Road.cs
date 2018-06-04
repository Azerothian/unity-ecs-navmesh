﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent (typeof (MeshCollider), typeof (NavMeshModifier))]
public class Road : MonoBehaviour {

	// Use this for initialization
	void Start () {
		gameObject.isStatic = true;
		var modifierRoad = GameObject.Find ("Road").GetComponent<NavMeshModifier> ();
		var modifier = gameObject.GetComponent<NavMeshModifier> ();
		modifier.overrideArea = modifierRoad.overrideArea;
		modifier.area = modifierRoad.area;
	}
}