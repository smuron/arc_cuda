import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const ws = new WebSocket('ws://192.168.1.86:8080');
let peerConnection;
let dataChannel;
let peerId;
let pendingCandidates = [];
let clearedTestData = false;
var simulationParams = {
   nozzle_origin: [0.0, 0.0, -2.5],
   nozzle_initial_acceleration: [0.0, 0.0, 0.0],
   nozzle_initial_velocity: [0.0, 0.0, 4.56],
   nozzle_velocity_spread: 0.4,
   nozzle_acceleration_spread: 0.0,
   nozzle_angle: 0.25,
   particle_start_temperature: 20000.0,
   particle_end_temperature: 9001.0,
minVoxelTemperature: 32,
};

ws.onopen = () => {
   console.log("ws onopen");
   peerConnection = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }] // Add STUN server
   });
   // Create a placeholder data channel to ensure SDP includes it
   const placeholderChannel = peerConnection.createDataChannel("placeholder");
   placeholderChannel.onopen = () => console.log("Placeholder channel opened (shouldn’t happen)");
   placeholderChannel.onclose = () => console.log("Placeholder channel closed");

   peerConnection.onicecandidate = (event) => {
      console.log("onicecandidate", event.candidate);
      if (event.candidate) {
         ws.send(JSON.stringify({
            type: 'ice_candidate',
            candidate: event.candidate.candidate, // Extract the candidate string
            peer_id: peerId
         }));
      }
   };

   peerConnection.oniceconnectionstatechange = () => {
      console.log("ICE connection state:", peerConnection.iceConnectionState);
   };

   // Create data channel (though server will create it, we set up listener)
   peerConnection.ondatachannel = (event) => {
      console.log("peerConnection creating data channel");
      dataChannel = event.channel;
      dataChannel.binaryType = 'arraybuffer';

      dataChannel.onmessage = (event) => {
         console.log("got data", event.data);
         const arrayBuffer = event.data;
         const view = new DataView(arrayBuffer);
         const type = view.getUint32(0, true);
         const size = view.getUint32(4, true);
         const totalElements = view.getUint32(8, true);

         if (!clearedTestData) {
            particlePositions.fill(0);
            particleVelocities.fill(0);
            voxelDensity.fill(0);
            voxelTemperature.fill(0);
            clearedTestData = true;
         }

         if (type === 1) { // PACKET_PARTICLES
            // Ensure the arrays are the right size
            if (totalElements !== numParticles) {
               numParticles = totalElements;
               particlePositions = new Float32Array(numParticles * 3);
               particleVelocities = new Float32Array(numParticles * 3);
               particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
               particleGeometry.setAttribute('velocity', new THREE.BufferAttribute(particleVelocities, 3));
            }

            // Update particle data
            let velocityMean = [0, 0, 0];
            for (let i = 0; i < totalElements; i++) {
               const offset = 12 + i * 12; // Header (12) + 3 floats (12) per particle
               const velocityOffset = offset + totalElements * 12;
               particlePositions[i * 3] = view.getFloat32(offset, true);     // pos.x
               particlePositions[i * 3 + 1] = view.getFloat32(offset + 4, true); // pos.y
               particlePositions[i * 3 + 2] = view.getFloat32(offset + 8, true); // pos.z
               velocityMean[0] += view.getFloat32(velocityOffset, true);
               velocityMean[1] += view.getFloat32(velocityOffset + 4, true);
               velocityMean[2] += view.getFloat32(velocityOffset + 8, true);
               particleVelocities[i * 3] = view.getFloat32(velocityOffset, true);   // vel.x
               particleVelocities[i * 3 + 1] = view.getFloat32(velocityOffset + 4, true); // vel.y
               particleVelocities[i * 3 + 2] = view.getFloat32(velocityOffset + 8, true); // vel.z
            }
            for (let i = 0; i < 3; i++) {
               velocityMean[i] /= totalElements;
            }

            // Mark attributes as needing update
            particleGeometry.attributes.position.needsUpdate = true;
            particleGeometry.attributes.velocity.needsUpdate = true;
            console.log('Updated particles:', totalElements, "vel mean", velocityMean);
         } else if (type === 2) { // PACKET_VOXELS
            for (let i = 0; i < totalElements; i++) {
               const offset = 12 + i * 5; // Header (12) + 5 bytes per voxel (x, y, z, density, temp)
               const x = view.getUint8(offset);
               const y = view.getUint8(offset + 1);
               const z = view.getUint8(offset + 2);
               const idx = x + y * voxelSize + z * voxelSize * voxelSize;

               // Validate coordinates
               if (x >= voxelSize || y >= voxelSize || z >= voxelSize) {
                  console.warn(`Invalid voxel coordinate: (${x}, ${y}, ${z})`);
                  continue;
               }

               voxelDensity[idx] = view.getUint8(offset + 3);
               voxelTemperature[idx] = view.getUint8(offset + 4);
            }

            // Regenerate surface voxels with the new data
            updateSurfaceVoxels();
            console.log('Updated voxels:', totalElements);
         }
      };
   };

   ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      console.log("websocket message", msg);
      if (msg.type === 'peer_id') {
         peerId = msg.id;
         peerConnection.createOffer()
            .then(offer => {
               console.log("Generated offer:", offer);
               return peerConnection.setLocalDescription(offer);
            })
            .then(() => {
               ws.send(JSON.stringify({
                  type: 'offer',
                  sdp: peerConnection.localDescription.sdp,
                  peer_id: peerId
               }));
            })
            .catch(err => console.error("Error creating offer:", err));
      } else if (msg.type === 'answer') {
         peerConnection.setRemoteDescription(new RTCSessionDescription(msg))
            .then(() => {
               // Process any pending ICE candidates
               while (pendingCandidates.length > 0) {
                  const candidate = pendingCandidates.shift();
                  peerConnection.addIceCandidate(new RTCIceCandidate(candidate))
                     .catch(err => console.error("Error adding queued ICE candidate:", err));
               }
            })
            .catch(err => console.error("Error setting remote description:", err));
      } else if (msg.type === 'ice_candidate') {
         const candidateObj = {
            candidate: msg.candidate,
            sdpMid: "0",
            sdpMLineIndex: 0
         };
         // If remote description isn’t set yet, queue the candidate
         if (!peerConnection.remoteDescription) {
            console.log("Queuing ICE candidate until remote description is set");
            pendingCandidates.push(candidateObj);
         } else {
            peerConnection.addIceCandidate(new RTCIceCandidate(candidateObj))
               .catch(err => console.error("Error adding ICE candidate:", err));
         }
      }
   }
};

function initializeUI() {
   document.getElementById('nozzle-origin-x').value = simulationParams.nozzle_origin[0];
   document.getElementById('nozzle-origin-y').value = simulationParams.nozzle_origin[1];
   document.getElementById('nozzle-origin-z').value = simulationParams.nozzle_origin[2];
   document.getElementById('nozzle-velocity-x').value = simulationParams.nozzle_initial_velocity[0];
   document.getElementById('nozzle-velocity-y').value = simulationParams.nozzle_initial_velocity[1];
   document.getElementById('nozzle-velocity-z').value = simulationParams.nozzle_initial_velocity[2];
   document.getElementById('nozzle-velocity-spread').value = simulationParams.nozzle_velocity_spread;
   document.getElementById('nozzle-angle').value = simulationParams.nozzle_angle;

   document.getElementById('particle-start-temperature').value = simulationParams.particle_start_temperature;
   document.getElementById('particle-end-temperature').value = simulationParams.particle_end_temperature;
}

// Update sendControlMessage to use simulationParams
function sendControlMessage() {
   if (!dataChannel || dataChannel.readyState !== "open") {
      console.warn("Data channel is not ready for control messages");
      return;
   }
   const message = JSON.stringify({
      type: 3, // CONTROL
      parameters: simulationParams
   });
   dataChannel.send(message);
   console.log("Sent control message:", simulationParams);
}
function sendResetMessage() {
   if (!dataChannel || dataChannel.readyState !== "open") {
      console.warn("Data channel is not ready for control messages");
      return;
   }
   const message = JSON.stringify({
      type: 4, // CONTROL
   });
   dataChannel.send(message);
   console.log("Sent control message:", simulationParams);
}

// Handle form submission
document.getElementById('param-form').addEventListener('submit', (event) => {
   event.preventDefault(); // Prevent page reload

   if (event.submitter.id == "reset") {
      sendResetMessage();
      return
   }

   // Update simulationParams with form values
   simulationParams.nozzle_origin[0] = parseFloat(document.getElementById('nozzle-origin-x').value);
   simulationParams.nozzle_origin[1] = parseFloat(document.getElementById('nozzle-origin-y').value);
   simulationParams.nozzle_origin[2] = parseFloat(document.getElementById('nozzle-origin-z').value);
   simulationParams.nozzle_initial_velocity[0] = parseFloat(document.getElementById('nozzle-velocity-x').value);
   simulationParams.nozzle_initial_velocity[1] = parseFloat(document.getElementById('nozzle-velocity-y').value);
   simulationParams.nozzle_initial_velocity[2] = parseFloat(document.getElementById('nozzle-velocity-z').value);
   simulationParams.nozzle_velocity_spread = parseFloat(document.getElementById('nozzle-velocity-spread').value);
   simulationParams.nozzle_angle = parseFloat(document.getElementById('nozzle-angle').value);
   simulationParams.particle_start_temperature = parseFloat(document.getElementById('particle-start-temperature').value);
   simulationParams.particle_end_temperature = parseFloat(document.getElementById('particle-end-temperature').value);
   updateNozzleMarker();
simulationParams.minVoxelTemperature = parseFloat(document.getElementById('min-voxel-temperature').value);

   // Send updated parameters via WebSocket
   sendControlMessage();
});

// Call this after DOM is ready (e.g., after renderer setup)
initializeUI();

var numParticles = 1000;
var voxelSize = 64;
var voxelSize3 = voxelSize * voxelSize * voxelSize;
var particlePositions = new Float32Array(numParticles * 3);
var particleVelocities = new Float32Array(numParticles * 3);
var voxelDensity = new Uint8Array(voxelSize * voxelSize * voxelSize);
var voxelTemperature = new Uint8Array(voxelSize * voxelSize * voxelSize);
const voxelPositions = new Float32Array(voxelSize3 * 3);

// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
const controls = new OrbitControls(camera, renderer.domElement);
document.body.appendChild(renderer.domElement);

// Convert particle data to geometry
const particleGeometry = new THREE.BufferGeometry();



const voxelOrigin = [0.0, 0.0, 0.0];
const voxelWorldSize = 1.0;
const voxelVertexShader = `
    attribute vec3 instancePosition;
    attribute float instanceDensity;
    attribute float instanceTemperature;
    varying vec3 vColor;
    varying vec2 vUv; // Pass UVs to fragment shader
    void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position + instancePosition, 1.0);
        float tempNormalized = instanceTemperature;
        //vColor = vec3(tempNormalized); // Grayscale
        vColor = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), tempNormalized); // Blue to red
        vUv = uv; // Pass UVs to fragment shader
        gl_Position = projectionMatrix * mvPosition;
    }
`;
const voxelFragmentShader = `
    varying vec3 vColor;
    varying vec2 vUv;
    void main() {
                gl_FragColor = vec4(vColor, 1.0);
    }
`;
const voxelMaterial = new THREE.ShaderMaterial({
   vertexShader: voxelVertexShader,
   fragmentShader: voxelFragmentShader,
   transparent: false,
});

// Surface extraction
let voxelMesh;
function updateSurfaceVoxels() {
   console.time('Surface Extraction');
   const surfaceVoxels = [];
   function isSurfaceVoxel(x, y, z) {
      const idx = x + y * voxelSize + z * voxelSize * voxelSize;
      if (voxelDensity[idx] <= 0) return false;
      if (voxelTemperature[idx] < simulationParams.minVoxelTemperature) return false;
      if (x === 0 || x === voxelSize - 1 || y === 0 || y === voxelSize - 1 || z === 0 || z === voxelSize - 1) return true;
      return (
         voxelDensity[idx - 1] <= 0 ||
         voxelDensity[idx + 1] <= 0 ||
         voxelDensity[idx - voxelSize] <= 0 ||
         voxelDensity[idx + voxelSize] <= 0 ||
         voxelDensity[idx - voxelSize * voxelSize] <= 0 ||
         voxelDensity[idx + voxelSize * voxelSize] <= 0
      );
   }

   for (let z = 0; z < voxelSize; z++) {
      for (let y = 0; y < voxelSize; y++) {
         for (let x = 0; x < voxelSize; x++) {
            if (isSurfaceVoxel(x, y, z)) {
               const idx = x + y * voxelSize + z * voxelSize * voxelSize;
               surfaceVoxels.push({
                  idx: idx,
                  x: (x / voxelSize - 0.5) * voxelWorldSize + voxelOrigin[0],
                  y: (y / voxelSize - 0.5) * voxelWorldSize + voxelOrigin[1],
                  z: (z / voxelSize - 0.5) * voxelWorldSize + voxelOrigin[2],
               });
            }
         }
      }
   }

   const instanceCount = surfaceVoxels.length;
   console.log(`Rendering ${instanceCount} surface voxels`);

   // Update or create the instanced mesh
   if (!voxelMesh || voxelMesh.count !== instanceCount) {
      if (voxelMesh) scene.remove(voxelMesh);
      const voxelScale = voxelWorldSize / voxelSize;
      const cubeGeometry = new THREE.BoxGeometry(voxelScale, voxelScale, voxelScale);
      voxelMesh = new THREE.InstancedMesh(cubeGeometry, voxelMaterial, instanceCount);
      scene.add(voxelMesh);
   }

   // Update attributes
   const surfaceVoxelPositions = new Float32Array(instanceCount * 3);
   const surfaceVoxelDensities = new Float32Array(instanceCount);
   const surfaceVoxelTemperatures = new Float32Array(instanceCount);
   for (let i = 0; i < instanceCount; i++) {
      const idx = surfaceVoxels[i].idx;
      surfaceVoxelPositions[i * 3] = surfaceVoxels[i].x;
      surfaceVoxelPositions[i * 3 + 1] = surfaceVoxels[i].y;
      surfaceVoxelPositions[i * 3 + 2] = surfaceVoxels[i].z;
      surfaceVoxelDensities[i] = voxelDensity[idx] / 255.0;
      surfaceVoxelTemperatures[i] = voxelTemperature[idx] / 255.0;
   }

   voxelMesh.geometry.setAttribute('instancePosition', new THREE.InstancedBufferAttribute(surfaceVoxelPositions, 3));
   voxelMesh.geometry.setAttribute('instanceDensity', new THREE.InstancedBufferAttribute(surfaceVoxelDensities, 1));
   voxelMesh.geometry.setAttribute('instanceTemperature', new THREE.InstancedBufferAttribute(surfaceVoxelTemperatures, 1));
   voxelMesh.instanceMatrix.needsUpdate = true;

   console.timeEnd('Surface Extraction');
}
// Fill the position array
for (let i = 0; i < numParticles; i++) {
   particlePositions[i] = (Math.random() - 0.5) * 10;     // x
   particlePositions[i + 1] = (Math.random() - 0.5) * 10; // y
   particlePositions[i + 2] = (Math.random() - 0.5) * 10; // z
   particleVelocities[i] = 0.0;
}
// Fill voxels with defaults
const randomTheta = Math.random() * 2.0 * Math.PI;
const randomMagnitude = Math.random() * 0.813;
let temperatureMean = 0;
for (let i = 0; i < voxelSize3; i++) {
   voxelDensity[i] = 255;
   voxelTemperature[i] = Math.floor(64
      + (64 * i % (voxelSize))
      + (64 * Math.sin(randomTheta + randomMagnitude * ((i / voxelSize) % voxelSize)))
      + 63 * Math.cos(randomTheta + randomMagnitude * 0.6 * (i / (voxelSize * voxelSize)))

   );
   temperatureMean += voxelTemperature[i];
}
temperatureMean /= voxelSize3;
console.log(voxelTemperature[0], voxelTemperature[voxelSize3 - 1], temperatureMean);

// Set the position attribute

// Vertex Shader
const particleVertexShader = `
  attribute vec3 velocity;
  varying vec3 vColor;
  
  void main() {
    vColor = vec3(0.91, 0.26, 0.44); // Base color
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(2.0, length(velocity) * 1.5); // Size based on velocity magnitude
    gl_Position = projectionMatrix * mvPosition;
  }
`;

// Fragment Shader
const particleFragmentShader = `
  varying vec3 vColor;
  
  void main() {
    gl_FragColor = vec4(vColor, 1.0);
  }
`;


// Create the shader material
const particleMaterial = new THREE.ShaderMaterial({
   vertexShader: particleVertexShader,
   fragmentShader: particleFragmentShader,
   transparent: true,
});
particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
particleGeometry.setAttribute('velocity', new THREE.BufferAttribute(particleVelocities, 3));
const particles = new THREE.Points(particleGeometry, particleMaterial);
scene.add(particles);

// After scene setup (after const controls = new OrbitControls...)
const nozzleMarkerGeometry = new THREE.SphereGeometry(0.05, 16, 16); // Small sphere, radius 0.05
const nozzleMarkerMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 }); // Yellow
const nozzleMarker = new THREE.Mesh(nozzleMarkerGeometry, nozzleMarkerMaterial);
scene.add(nozzleMarker);

// Function to update nozzle marker position
function updateNozzleMarker() {
   nozzleMarker.position.set(
      simulationParams.nozzle_origin[0],
      simulationParams.nozzle_origin[1],
      simulationParams.nozzle_origin[2]
   );
}

// Call this initially to set the starting position
updateNozzleMarker();

// Camera position
camera.position.x = 1;
camera.position.y = 1;
camera.position.z = 5;

updateSurfaceVoxels();

// Animation loop
function animate() {
   requestAnimationFrame(animate);

   // Optional: Update particle positions if they're dynamic
   // const positions = geometry.attributes.position.array;
   // for (let i = 0; i < positions.length; i += 3) {
   //   positions[i] += Math.random() * 0.1 - 0.05;     // x
   //   positions[i + 1] += Math.random() * 0.1 - 0.05; // y
   //   positions[i + 2] += Math.random() * 0.1 - 0.05; // z
   // }
   // geometry.attributes.position.needsUpdate = true;

   renderer.render(scene, camera);
}
animate();

// Handle window resize
window.addEventListener('resize', () => {
   camera.aspect = window.innerWidth / window.innerHeight;
   camera.updateProjectionMatrix();
   renderer.setSize(window.innerWidth, window.innerHeight);
});
