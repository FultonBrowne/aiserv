# AI Serv

A powerful, flexible, and simple LLM server powered by llamaCPP

## Modules

- shurbai: The core AI module, responsible for all AI-related tasks
- api: A very effiecient API powered by shurbai

## Usage

to run the server, create a config.json based on config.default.json and run the following commands:

```bash
cd api
cargo build --release
cd ..
./target/release/api
```

You must run the server in the same directory as the config.json file, that will change in the future

## History

This Server was built as an intern project and a candidate for client deployment during my time at Shurburt LLC. It was built under the supervision of (and for other, components, in collaboration with) Kevin Auer, Jay North, Kelsey Weeks, and Ian Harrison. It is now being run as its own little hobby project under the GPL-3.0 License.
