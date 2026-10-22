const net = require("node:net");
if (process.env.APS_NODE_NETWORK_GUARD !== "1") process.exit(2);
try {
  net.connect(443, "203.0.113.1");
  process.exit(3);
} catch (e) {
  if (!e.message.includes("Offline test")) throw e;
  console.log("Node network denial verified");
  process.exit(0);
}
