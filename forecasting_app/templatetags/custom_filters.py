from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    if dictionary is None:
        return None
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    try:
        return getattr(dictionary, key)
    except AttributeError:
        return None