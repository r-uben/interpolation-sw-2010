import boto3
import pandas as pd
import json

class S3:
    """AWS S3 and Secrets Manager utility class."""

    @staticmethod
    def get_buckets():
        """List all S3 buckets."""
        s3 = boto3.client('s3')
        buckets = s3.list_buckets().get('Buckets', [])
        return pd.DataFrame([bucket['Name'] for bucket in buckets], columns=['name'])
    
    @staticmethod
    def list_secrets():
        """List all secrets in AWS Secrets Manager."""
        client = boto3.client('secretsmanager')
        response = client.list_secrets()
        if 'SecretList' in response:
            secrets = [secret['Name'] for secret in response['SecretList']]
            return secrets
        else:
            return []
        
    @staticmethod
    def get_secret(secret_name, key=None):
        """Get a secret from AWS Secrets Manager.
        
        Args:
            secret_name (str): Name of the secret
            key (str, optional): If the secret is a JSON object, get this specific key
            
        Returns:
            str: The secret value, or None if not found
        """
        client = boto3.client('secretsmanager')
        try:
            response = client.get_secret_value(SecretId=secret_name)
            if 'SecretString' in response:
                secret_value = response['SecretString']
                # If key is provided, parse JSON and return specific key
                if key:
                    try:
                        secret_dict = json.loads(secret_value)
                        return secret_dict.get(key)
                    except json.JSONDecodeError:
                        return None
                return secret_value
            else:
                return None
        except Exception as e:
            print(f"Error retrieving secret {secret_name}: {e}")
            return None

    @staticmethod
    def store_secret(secret_name, token, password, type="api"):
        """Store a secret in AWS Secrets Manager.
        
        Args:
            secret_name (str): Name of the secret
            token (str): API token or username
            password (str): API key or password
            type (str): Type of secret ('api' or 'password')
            
        Returns:
            str: ARN of the created secret, or None if failed
        """
        client = boto3.client('secretsmanager')
        try:
            if type == "api":
                secret_value = {
                    'api_token': token,
                    'api_key': password
                }
            elif type == "password":
                secret_value = {
                    'username': token,
                    'password': password,
                }
            else:
                raise ValueError(f"Invalid type {type}")
            response = client.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value)
            )
            return response['ARN']
        except client.exceptions.ResourceExistsException:
            print(f"Secret {secret_name} already exists. Use update_secret method to modify it.")
            return None
        except Exception as e:
            print(f"Error storing secret {secret_name}: {e}")
            return None
