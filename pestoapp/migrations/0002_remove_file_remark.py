# Generated by Django 3.0.3 on 2020-04-08 09:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pestoapp', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='file',
            name='remark',
        ),
    ]