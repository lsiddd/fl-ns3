"""
Main entry point for Federated Learning API Server.
"""

import argparse
from api import FLAPIServer


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Federated Learning API Server")
    parser.add_argument('--port', type=int, default=5000, 
                       help='Starting port for the server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args, _ = parser.parse_known_args()
    
    # Create and run server
    server = FLAPIServer()
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()