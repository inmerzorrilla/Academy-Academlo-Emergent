import React from 'react';

export const SocialLinks = () => {
  const socialLinks = [
    {
      name: 'Facebook',
      icon: 'fab fa-facebook-f',
      url: 'https://www.facebook.com/emergent.academlo.academy/',
      color: 'text-blue-500 border-blue-500 hover:bg-blue-500'
    },
    {
      name: 'YouTube',
      icon: 'fab fa-youtube',
      url: 'https://www.youtube.com/channel/UCWIDXeEijD1wAvba-2Xg-yA',
      color: 'text-red-500 border-red-500 hover:bg-red-500'
    },
    {
      name: 'LinkedIn',
      icon: 'fab fa-linkedin-in',
      url: 'https://www.linkedin.com/in/emergent-academlo-academy-0037a3386/',
      color: 'text-blue-600 border-blue-600 hover:bg-blue-600'
    }
  ];

  return (
    <div className="social-links">
      {socialLinks.map((link) => (
        <a
          key={link.name}
          href={link.url}
          target="_blank"
          rel="noopener noreferrer"
          className={`social-link ${link.color}`}
          data-testid={`social-${link.name.toLowerCase()}`}
          title={link.name}
        >
          <i className={link.icon}></i>
        </a>
      ))}
    </div>
  );
};
