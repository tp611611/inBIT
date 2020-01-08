package com.tutorialspoint;
import java.io.*;
public class ReName {
	public static String [] getFileName(String path)
    {
        File file = new File(path);
        String [] fileName = file.list();
        return fileName;
    }
    public static void renameFile(String path,String oldname,String newname){ 
        if(!oldname.equals(newname)){//�µ��ļ�������ǰ�ļ�����ͬʱ,���б�Ҫ���������� 
            File oldfile=new File(path+"\\"+oldname); 
            File newfile=new File(path+"\\"+newname); 
            if(!oldfile.exists()){
                return;//�������ļ�������
            }
            if(newfile.exists())//���ڸ�Ŀ¼���Ѿ���һ���ļ������ļ�����ͬ�������������� 
                System.out.println(newname+"�Ѿ����ڣ�"); 
            else{ 
                oldfile.renameTo(newfile); 
                System.out.println(oldfile+"�Ѿ��޸�Ϊ" + newfile);
            } 
        }else{
            System.out.println("���ļ����;��ļ�����ͬ...");
        }
    }
    public static void main(String[] args)
    {
        String [] fileName = getFileName("F:\\����\\homework\\ϵͳ��ѧԭ��\\����\\�п����·���");//<span style="font-family: Arial, Helvetica, sans-serif;">�˴��޸�Ϊ��ı���·��</span>
        for (int i = 0; i < fileName.length; i++) {
			renameFile("F:\\����\\homework\\ϵͳ��ѧԭ��\\����\\�п����·���", fileName[i], "- "+i+ " "+fileName[i]);//cx�޸�Ϊ��Ҫ�޸ĵ��ļ�����ʽ
		}
    }

}
