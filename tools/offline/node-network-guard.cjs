// Test-process network policy; inherited by Node child processes and workers.
const net = require('node:net');
const dgram = require('node:dgram');
const dns = require('node:dns');
const loopback = host => host === 'localhost' || host === '::1' || host === '127.0.0.1' || host === '::ffff:127.0.0.1';
function check(host) {
  if (!loopback(host)) throw new Error(`Offline test attempted external network: ${host}`);
}
for (const name of Object.keys(process.env)) {
  if (/(API_KEY|ACCESS_TOKEN|AUTH_TOKEN)$|^(OPENAI_ADMIN_KEY|HF_TOKEN)$/.test(name)) delete process.env[name];
}
const connect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  let options = Array.isArray(args[0]) ? args[0][0] : args[0];
  if (typeof options === 'object' && options !== null) {
    if (!options.path) check(options.host || 'localhost');
  } else if (typeof options === 'number') {
    check(typeof args[1] === 'string' ? args[1] : 'localhost');
  } else if (typeof options === 'string' && /^\d+$/.test(options)) {
    check(typeof args[1] === 'string' ? args[1] : 'localhost');
  } else if (typeof options !== 'string') {
    throw new Error('Unknown socket signature under offline guard');
  }
  return connect.apply(this, args);
};
const lookup = dns.lookup;
dns.lookup = function (host, ...args) { check(host); return lookup.call(this, host, ...args); };
const send = dgram.Socket.prototype.send;
dgram.Socket.prototype.send = function (...args) {
  const host = args.findLast(value => typeof value === 'string');
  check(host);
  return send.apply(this, args);
};
process.env.APS_NODE_NETWORK_GUARD = '1';
